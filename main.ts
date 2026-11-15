// free-model-aggregator-02 entry: one server, two client wires, one live free model
// behind every answer.
//
// The server reads the trial brief (the free model list with the house's ratings and
// the six prompts' manifest), registers with the env, and answers each prompt by
// dialling a real free OpenRouter model through the grader's gateway on the trial's
// key. Nothing here is simulated: every completion comes from the gateway, and the
// `model` field on every answer names the model that really served it.
//
// ROUTING (quality is 60% of the score): for each prompt, the candidates are the
// brief's free models that meet the prompt's requirements (tools / vision / JSON
// mode / context), ranked by rating_milli, highest first, brief order breaking
// ties. The best model serves first.
//
// FAILOVER (10% of the score): a gateway 429 or 5xx (or a dropped connection) is
// not an answer. The prompt is retried on the next-ranked candidate that meets it,
// up to three attempts a prompt, and every refusal is reported on the served
// response in the `x-makefun-upstream-failures` header (at most 8 rows), so the
// failover column can see a refusal the server recovered from. A model that
// refused is cooled down for the rest of the trial: later prompts never spend a
// call on it, so one rate-limited model cannot take every later prompt down.
//
// DETERMINISM: every choice is a function of the brief, the requests and the
// gateway's answers so far. No wall clock and no unseeded randomness anywhere in
// a request path, so a replay leaves the same state export byte for byte.
//
// WIRING: the entry dials every model on the gateway's Chat Completions route
// (POST $OPENAI_BASE_URL/chat/completions). An Anthropic-shaped request is
// translated into a Chat Completions body first and the answer translated back.
// The gateway refuses a request with no output cap, so every upstream body gets a
// max_tokens (the client's, clamped to the 8192 cap, or 1024 when it named none).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway's per-request output cap; a request over it is refused. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap sent upstream when the client named none. */
const DEFAULT_OUTPUT_TOKENS = 1024;
/** How many ranked candidates one prompt is tried on before it is given up on. */
const MAX_ATTEMPTS_PER_PROMPT = 3;
/** How long one gateway call may take; the env's client waits at most 30 s. */
const GATEWAY_TIMEOUT_MS = 25000;
/** The most refusal rows the failure-trail header may carry. */
const MAX_FAILURE_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Models that refused a call earlier in this trial: never spent a call on again. */
const cooledDown = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible
 * root (`…/v1` on the trial network); a base that does not already end in `/v1`
 * gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The ranked candidates for one prompt: the brief's free models that meet its
 * requirements, best rating first, the brief's order breaking ties. A cooled-down
 * model is left out: it refused earlier in this trial, and a retry on it would
 * spend a gateway call the tie-break counts.
 */
function rankedCandidates(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => !cooledDown.has(model.id) && meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One refusal row for the `x-makefun-upstream-failures` trail header. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The gateway's error code for a refusal, from its body when it names one. */
function failureCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.code === 'string' && parsed.error.code !== '') {
      return parsed.error.code;
    }
    if (isRecord(parsed) && typeof parsed.code === 'string' && parsed.code !== '') return parsed.code;
  } catch {
    // Not JSON: fall through to the status-derived code.
  }
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'server_error';
  if (status === 401) return 'unauthorized';
  if (status === 400) return 'invalid_request_error';
  return 'upstream_error';
}

/**
 * An upstream body with an output cap the gateway accepts: the client's
 * max_tokens (or max_completion_tokens), clamped to the 8192 cap, or the
 * default when it named none. A request with no cap is refused outright.
 */
function withOutputCap(body: Json): Json {
  const named = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  const capped = named === undefined ? DEFAULT_OUTPUT_TOKENS : Math.max(1, Math.min(MAX_OUTPUT_TOKENS, Math.floor(named)));
  return { ...body, max_tokens: capped };
}

/** One call to the gateway's Chat Completions route, naming the model dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...withOutputCap(body), model }),
    signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
  });
}

/**
 * Dial the ranked candidates for one prompt until one answers. Returns the
 * answering response with the model that served it and the refusal trail, or
 * the trail alone when every attempt was refused and the prompt is lost. A
 * refusal cools its model down for the rest of the trial.
 */
async function serveWithFailover(
  candidates: ModelRow[],
  buildBody: (model: string) => Json,
): Promise<{ upstream: Response | null; model: string | null; failures: FailureRow[] }> {
  const failures: FailureRow[] = [];
  const attempts = candidates.slice(0, MAX_ATTEMPTS_PER_PROMPT);
  for (const candidate of attempts) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, buildBody(candidate.id));
    } catch (err) {
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      cooledDown.add(candidate.id);
      log(`dial ${candidate.id} failed to reach the gateway: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { upstream, model: candidate.id, failures };
    const text = await upstream.text();
    failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: candidate.id });
    cooledDown.add(candidate.id);
    log(`dial ${candidate.id} refused HTTP ${upstream.status}`);
  }
  return { upstream: null, model: null, failures };
}

/** The refusal trail as one header value, or null when there is nothing to report. */
function failuresHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS)) };
}

/** A prompt no candidate could serve, as an error in the client's wire. */
function lost(res: ServerResponse, wire: 'openai' | 'anthropic', failures: FailureRow[]): void {
  const last = failures[failures.length - 1];
  const detail = last === undefined ? 'no free model in the brief meets this request' : `the gateway refused every model tried (last HTTP ${last.status}: ${last.code})`;
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', detail);
  else sendError(res, wire, 502, 'upstream_error', detail);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = rankedCandidates(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const served = await serveWithFailover(candidates, (model) => ({ ...body, model }));
  if (served.upstream === null || served.model === null) return lost(res, 'openai', served.failures);
  const { upstream, model, failures } = served;
  const trail = failuresHeader(failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
    try {
      for await (const frame of openAIStreamWithModel(upstream.body as ReadableStream<Uint8Array>, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), trail);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = rankedCandidates(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const served = await serveWithFailover(candidates, (model) => anthropicToOpenAI(body, model));
  if (served.upstream === null || served.model === null) return lost(res, 'anthropic', served.failures);
  const { upstream, model, failures } = served;
  const trail = failuresHeader(failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body as ReadableStream<Uint8Array>, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
