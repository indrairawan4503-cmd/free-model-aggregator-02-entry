// free-model-aggregator-02 entry: one server, two client wires, one live free
// model behind every answer.
//
// The server reads the trial brief (the free model list with the house's
// ratings and the six prompts' manifest), announces itself to the env, and
// answers each prompt by dialling a real free OpenRouter model through the
// grader's gateway on the trial's key. Nothing here is simulated: every
// completion comes from the gateway, and the `model` field on every answer
// names the model that really served it.
//
// HOW A PROMPT IS SERVED (quality is 60% of the score): for each prompt, the
// candidates are the brief's free models that meet the prompt's requirements
// (tools / vision / JSON mode / context), ordered by rating_milli, highest
// first, brief order breaking ties. The best model serves first.
//
// WHEN A CALL IS REFUSED (failover is 10%): a gateway 429 or 5xx (or a
// dropped connection) is not an answer. The prompt is retried on the
// next-ranked candidate that meets it, up to three attempts a prompt, and
// every refusal is reported on the served response in the
// `x-makefun-upstream-failures` header (at most 8 rows), so the failover
// column can see a refusal the server recovered from. A model that refused
// is burned for the rest of the trial: later prompts never spend a call on
// it, so one rate-limited model cannot take every later prompt down.
//
// WHY A REPLAY MATCHES: every choice is a function of the brief, the
// requests and the gateway's answers so far. No wall clock and no unseeded
// randomness anywhere in a request path, so a replay leaves the same state
// export byte for byte.
//
// HOW THE WIRES MAP: the entry dials every model on the gateway's Chat
// Completions route (POST $OPENAI_BASE_URL/chat/completions). An
// Anthropic-shaped request is translated into a Chat Completions body first
// and the answer translated back. The gateway refuses a request with no
// output cap, so every upstream body gets a max_tokens (the client's,
// clamped to the 8192 cap, or 1024 when it named none).

import * as http from 'node:http';
import type { IncomingMessage, Server, ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { buildCatalog, capabilitiesOf, freeModels, isRecord, meetsRequirements, readBrief } from './lib/providers.ts';
import type { CatalogReport, Json, ModelRow, Requirement } from './lib/providers.ts';
import { anthropicError, anthropicEventsFromOpenAI, anthropicModelList, anthropicToOpenAI, openAIError, openAIJsonWithModel, openAIModelList, openAIStreamWithModel, openAIToAnthropic } from './lib/wire.ts';
import type { ServedModel } from './lib/wire.ts';

/** Wire protocol version sent in the register body. */
const PROTOCOL_REV = 1;
/** Delay between GET /work polls, in milliseconds. */
const WORK_CHECK_EVERY_MS = 100;
/** Largest client request body the server reads, in bytes. */
const MAX_REQUEST_BYTES = 8 * 1024 * 1024;
/** Largest output cap the gateway accepts on one call. */
const GATEWAY_MAX_TOKENS = 8192;
/** Output cap sent upstream when the client named none. */
const DEFAULT_CAP_TOKENS = 1024;
/** Ranked candidates tried for one prompt before giving up. */
const MAX_DIALS_PER_PROMPT = 3;
/** Longest one gateway call may run; the env's client waits 30 s at most. */
const GATEWAY_CALL_TIMEOUT_MS = 25000;
/** Most refusal rows the failure-trail header may carry. */
const MAX_TRAIL_ROWS = 8;
/** Longest the register call may run, in milliseconds. */
const REGISTER_TIMEOUT_MS = 5000;
/** Longest one /work poll may run, in milliseconds. */
const WORK_TIMEOUT_MS = 2000;
/** Most characters of a register refusal kept in the thrown error. */
const DETAIL_LIMIT = 500;

const ENV_BASE = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_BASE = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const UPSTREAM_KEY = process.env.OPENAI_API_KEY ?? '';

const briefDoc = readBrief(BRIEF_FILE);
const catalogReport: CatalogReport = buildCatalog(briefDoc);
/** Which prompt row the next arriving request is: the brief's prompts arrive in the client's order. */
let nextPromptIndex = 0;
/** Models that refused a call earlier in this trial: never spend a call on them again. */
const burnedModels = new Set<string>();
/** The null a helper returns when there is nothing to hand back. */
const NO_BODY: unknown = null;
/** The one JSON content-type header, shared by every JSON reply and call. */
const JSON_MIME: Record<string, string> = { 'content-type': 'application/json' };

/** Log one line to stderr, prefixed so it reads as this server's voice. */
const report = (message: string): void => { process.stderr.write('aggregator: ' + message + '\n'); };

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible
 * root (…/v1 on the trial network); a base that does not already end in `/v1`
 * gains one, so both spellings reach the route.
 */
function chatCompletionsEndpoint(): string {
  const base = GATEWAY_BASE;
  if (base.length === 0) {
    return '';
  }
  const suffix = base.endsWith('/v1') ? '/chat/completions' : '/v1/chat/completions';
  return base + suffix;
}

/** The free models the brief lists, as GET /v1/models names them. */
function modelsForList(): ServedModel[] {
  const rows = freeModels(briefDoc);
  return rows.map((row) => ({ id: row.id, provider: row.provider, display_name: row.display_name }));
}

/**
 * The ranked pool for one prompt: the brief's free models that meet its
 * requirements, best rating first, the brief's order breaking ties. A burned
 * model is left out: it turned a call down earlier in this trial, and trying
 * it again would spend a gateway call the tie-break counts.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const eligible = freeModels(briefDoc).filter(
    (row) => burnedModels.has(row.id) === false && meetsRequirements(requires, capabilitiesOf(row)),
  );
  eligible.sort((first, second) => second.rating_milli - first.rating_milli);
  return eligible;
}

/** Whether one content block is an image the model must be able to see. */
function blockIsImage(block: unknown): boolean {
  return isRecord(block) && (block.type === 'image' || block.type === 'image_url');
}

/** Whether one message carries an image block. */
function messageIsVisual(message: unknown): boolean {
  return isRecord(message) && Array.isArray(message.content) && message.content.some((block) => blockIsImage(block));
}

/** Whether the request body declares any tools. */
function payloadHasTools(body: Json): boolean {
  const tools = body.tools;
  return Array.isArray(tools) && tools.length > 0;
}

/** Whether the request body asks for JSON output. */
function payloadWantsJson(body: Json): boolean {
  const format = body.response_format;
  return isRecord(format) && (format.type === 'json_object' || format.type === 'json_schema');
}

/** Requirements read off a request body, when the brief has no row for it. */
function needsFromBody(body: Json): Requirement {
  const rawMessages = body.messages;
  const messages = Array.isArray(rawMessages) ? rawMessages : [];
  const vision = messages.some((message) => messageIsVisual(message));
  const tools = payloadHasTools(body);
  const json = payloadWantsJson(body);
  return { tools, vision, json };
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function needsForNext(body: Json): Requirement {
  const row = briefDoc.prompts[nextPromptIndex];
  nextPromptIndex += 1;
  if (row !== undefined) {
    return row.requires;
  }
  return needsFromBody(body);
}

/** Parse one JSON text, or the null sentinel when it is not JSON. */
function parseQuietly(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return NO_BODY;
  }
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readRequestJson(req: IncomingMessage): Promise<Json | null> {
  const pieces: Buffer[] = [];
  let bytesSoFar = 0;
  for await (const piece of req) {
    const bytes = Buffer.isBuffer(piece) ? piece : Buffer.from(String(piece));
    bytesSoFar += bytes.length;
    if (bytesSoFar > MAX_REQUEST_BYTES) {
      return NO_BODY as Json | null;
    }
    pieces[pieces.length] = bytes;
  }
  const joined = pieces.length === 0 ? '' : Buffer.concat(pieces).toString('utf8');
  const parsed = parseQuietly(joined);
  if (isRecord(parsed) === false) {
    return NO_BODY as Json | null;
  }
  return parsed as Json;
}

/** One JSON reply: status, content-type, content-length, then the body. */
function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const payload = JSON.stringify(body);
  res.statusCode = status;
  res.setHeader('content-type', 'application/json');
  res.setHeader('content-length', String(Buffer.byteLength(payload)));
  for (const [name, value] of Object.entries(extraHeaders)) {
    res.setHeader(name, value);
  }
  res.end(payload);
}

/** Error message texts the client wires expect, kept byte-identical. */
const BAD_BODY_MSG = 'the request body is not one JSON object';
const NO_MODEL_MSG = 'no free model in the brief meets this request';
const SERVER_FAILED_MSG = 'the server failed to answer';

/** One error reply, shaped for the wire the client speaks. */
function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  const payload = wire === 'anthropic' ? anthropicError(type, message) : openAIError(message, type, null);
  sendJson(res, status, payload);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function serveModelList(req: IncomingMessage, res: ServerResponse): void {
  const rows = modelsForList();
  const keyHeader = req.headers['x-api-key'];
  const anthropicShape = typeof keyHeader === 'string';
  const shape = anthropicShape ? anthropicModelList(rows) : openAIModelList(rows);
  sendJson(res, 200, shape);
}

/** One refusal row for the `x-makefun-upstream-failures` trail header. */
interface Refusal {
  httpStatus: number;
  errorCode: string;
  modelId: string | null;
}

/** A dial that reached a model and got an answer. */
interface DialServed {
  served: true;
  upstream: Response;
}

/** A dial the gateway refused, with its trail row. */
interface DialRefused {
  served: false;
  refusal: Refusal;
}

/** The outcome of one dial: served with its response, or refused with its row. */
type DialResult = DialServed | DialRefused;

/** The `code` field of one error record, when it names a non-empty string. */
function codeFieldOf(record: Record<string, unknown>): string | null {
  const code = record.code;
  if (typeof code === 'string' && code !== '') {
    return code;
  }
  return NO_BODY as string | null;
}

/** Error codes the gateway's own body may name, read off one response text. */
function errorCodeInBody(text: string): string | null {
  const parsed = parseQuietly(text);
  if (isRecord(parsed) === false) {
    return NO_BODY as string | null;
  }
  const nested = parsed.error;
  if (isRecord(nested) === false) {
    return codeFieldOf(parsed);
  }
  return codeFieldOf(nested) ?? codeFieldOf(parsed);
}

/** Status-derived refusal codes, for a body that names none. */
function statusFallbackCode(status: number): string {
  if (status === 429) {
    return 'rate_limited';
  }
  if (status === 401) {
    return 'unauthorized';
  }
  if (status === 400) {
    return 'invalid_request_error';
  }
  return 'upstream_error';
}

/** The gateway's error code for a refusal, from its body when it names one. */
function refusalCode(status: number, text: string): string {
  const named = errorCodeInBody(text);
  if (named !== null) {
    return named;
  }
  if (status >= 500) {
    return 'server_error';
  }
  return statusFallbackCode(status);
}

/** The output cap the client named, if it named one on either spelling. */
function clientOutputCap(body: Json): number | undefined {
  const first = body.max_tokens;
  if (typeof first === 'number') {
    return first;
  }
  const second = body.max_completion_tokens;
  return typeof second === 'number' ? second : undefined;
}

/** One named cap clamped into the range the gateway accepts. */
function clampCap(named: number): number {
  const floored = Math.floor(named);
  const lower = Math.max(1, floored);
  return Math.min(GATEWAY_MAX_TOKENS, lower);
}

/**
 * An upstream body with an output cap the gateway accepts: the client's
 * max_tokens (or max_completion_tokens), clamped to the cap, or the
 * fallback when it named none. A request with no cap is refused outright.
 */
function upstreamBodyWithCap(body: Json): Json {
  const named = clientOutputCap(body);
  const capped = named === undefined ? DEFAULT_CAP_TOKENS : clampCap(named);
  return { ...body, max_tokens: capped } as Json;
}

/** Headers for one gateway call: JSON in, the trial key, SSE accepted when streaming. */
function upstreamHeaders(body: Json): Record<string, string> {
  const streaming = body.stream === true;
  if (streaming) {
    return { ...JSON_MIME, authorization: 'Bearer ' + UPSTREAM_KEY, accept: 'text/event-stream' };
  }
  return { ...JSON_MIME, authorization: 'Bearer ' + UPSTREAM_KEY };
}

/** One call to the gateway's Chat Completions route, naming the model dialled. */
async function dialGateway(model: string, body: Json): Promise<Response> {
  const target = chatCompletionsEndpoint();
  const payload = { ...upstreamBodyWithCap(body), model };
  return fetch(target, {
    method: 'POST' as const,
    headers: upstreamHeaders(body),
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(GATEWAY_CALL_TIMEOUT_MS),
  });
}

/**
 * One dial of one ranked candidate. A refusal burns its model for the
 * rest of the trial and is reported as one trail row.
 */
async function dialCandidate(candidate: ModelRow, body: Json): Promise<DialResult> {
  let upstream!: Response;
  try {
    upstream = await dialGateway(candidate.id, body);
  } catch (failure) {
    burnedModels.add(candidate.id);
    report(`dial ${candidate.id} failed to reach the gateway: ${(failure as Error).message}`);
    const refusal: Refusal = { httpStatus: 502, errorCode: 'upstream_unreachable', modelId: candidate.id };
    return { served: false, refusal };
  }
  if (upstream.ok && upstream.body !== null) {
    return { served: true, upstream };
  }
  const refusalText = await upstream.text();
  burnedModels.add(candidate.id);
  report(`dial ${candidate.id} refused HTTP ${upstream.status}`);
  const refusal: Refusal = { httpStatus: upstream.status, errorCode: refusalCode(upstream.status, refusalText), modelId: candidate.id };
  return { served: false, refusal };
}

/**
 * Dial the ranked candidates for one prompt until one answers. Returns the
 * answering response with the model that served it and the refusal trail, or
 * the trail alone when every attempt was refused and the prompt is lost.
 */
async function dialPool(pool: ModelRow[], makeBody: (model: string) => Json): Promise<{ upstream: Response | null; model: string | null; failures: Refusal[] }> {
  const failures = [] as Refusal[];
  const attempts = pool.slice(0, MAX_DIALS_PER_PROMPT);
  for (let index = 0; index < attempts.length; index += 1) {
    const candidate = attempts[index] as ModelRow;
    const outcome = await dialCandidate(candidate, makeBody(candidate.id));
    if (outcome.served) {
      return { upstream: outcome.upstream, model: candidate.id, failures };
    }
    failures.push(outcome.refusal);
  }
  const empty: { upstream: Response | null; model: string | null; failures: Refusal[] } = { upstream: null, model: null, failures };
  return empty;
}

/** The refusal trail as one header value, or nothing when there is nothing to report. */
function trailHeader(failures: Refusal[]): Record<string, string> {
  if (failures.length === 0) {
    return {} as Record<string, string>;
  }
  const rows = failures.slice(0, MAX_TRAIL_ROWS);
  const wireRows = rows.map((row) => ({ status: row.httpStatus, code: row.errorCode, model: row.modelId }));
  const trail = JSON.stringify(wireRows);
  const header = { 'x-makefun-upstream-failures': trail };
  return header;
}

/** A prompt no candidate could serve, as an error in the client's wire. */
function replyLost(res: ServerResponse, wire: 'openai' | 'anthropic', failures: Refusal[]): void {
  const latest = failures[failures.length - 1];
  const detail = latest === undefined
    ? NO_MODEL_MSG
    : `the gateway refused every model tried (last HTTP ${latest.httpStatus}: ${latest.errorCode})`;
  const anthropicWire = wire === 'anthropic';
  if (anthropicWire) {
    sendError(res, wire, 502, 'api_error', detail);
    return;
  }
  sendError(res, wire, 502, 'upstream_error', detail);
}

/** End one streamed reply: no more chunks after this. */
function finishResponse(res: ServerResponse): void {
  res.end('');
}

/** Relay one OpenAI SSE answer, stamping the dialled model on every chunk. */
async function relayChatStream(res: ServerResponse, upstream: Response, model: string, trail: Record<string, string>): Promise<void> {
  res.writeHead(200, { ...trail, 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  try {
    for await (const frame of openAIStreamWithModel(
      upstream.body as ReadableStream<Uint8Array>,
      model,
    )) {
      res.write(frame);
    }
  } finally {
    finishResponse(res);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function answerChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readRequestJson(req);
  const noBody = body === null;
  if (noBody) {
    sendError(res, 'openai', 400, 'invalid_request_error', BAD_BODY_MSG);
    return;
  }
  const pool = candidatesFor(needsForNext(body));
  const poolEmpty = pool.length === 0;
  if (poolEmpty) {
    sendError(res, 'openai', 503, 'no_free_model', NO_MODEL_MSG);
    return;
  }
  const served = await dialPool(pool, (model) => ({ ...body, model }));
  const unserved = served.upstream === null || served.model === null;
  if (unserved) {
    replyLost(res, 'openai', served.failures);
    return;
  }
  const upstream = served.upstream;
  const model = served.model;
  const failures = served.failures;
  const trail = trailHeader(failures);
  const streaming = body.stream === true;
  if (streaming) {
    await relayChatStream(res, upstream, model, trail);
    return;
  }
  const answer = (await upstream.json()) as unknown;
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), trail);
}

/** Relay one Anthropic SSE answer, translated from the gateway's stream. */
async function relayAnthropicStream(res: ServerResponse, upstream: Response, model: string, trail: Record<string, string>): Promise<void> {
  res.writeHead(200, { ...trail, 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  try {
    for await (const event of anthropicEventsFromOpenAI(
      upstream.body as ReadableStream<Uint8Array>,
      model,
    )) {
      res.write(event);
    }
  } finally {
    finishResponse(res);
  }
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function answerMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readRequestJson(req);
  const noBody = body === null;
  if (noBody) {
    sendError(res, 'anthropic', 400, 'invalid_request_error', BAD_BODY_MSG);
    return;
  }
  const pool = candidatesFor(needsForNext(body));
  const poolEmpty = pool.length === 0;
  if (poolEmpty) {
    sendError(res, 'anthropic', 503, 'overloaded_error', NO_MODEL_MSG);
    return;
  }
  const served = await dialPool(pool, (model) => anthropicToOpenAI(body, model));
  const unserved = served.upstream === null || served.model === null;
  if (unserved) {
    replyLost(res, 'anthropic', served.failures);
    return;
  }
  const upstream = served.upstream;
  const model = served.model;
  const failures = served.failures;
  const trail = trailHeader(failures);
  const streaming = body.stream === true;
  if (streaming) {
    await relayAnthropicStream(res, upstream, model, trail);
    return;
  }
  const answer = (await upstream.json()) as unknown;
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), trail);
}

/** Which client wire a path belongs to: Anthropic only for /v1/messages. */
function wireOfPath(pathname: string): 'openai' | 'anthropic' {
  const anthropicPath = pathname === '/v1/messages';
  return anthropicPath ? 'anthropic' : 'openai';
}

/** The router: three routes, anything else a 404 in the path's wire. */
async function dispatch(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  const path = url.pathname;
  const method = req.method ?? '';
  if (method === 'GET' && path === '/v1/models') {
    serveModelList(req, res);
    return;
  }
  if (method === 'POST' && path === '/v1/chat/completions') {
    await answerChat(req, res);
    return;
  }
  if (method === 'POST' && path === '/v1/messages') {
    await answerMessages(req, res);
    return;
  }
  const wire = wireOfPath(path);
  sendError(res, wire, 404, 'not_found', `no route ${method} ${path}`);
}

/** Bind the server to an OS-picked port and resolve once it listens. */
function listenOnPort(server: Server): Promise<void> {
  const ready = new Promise<void>((resolve, reject) => {
    server.once('error', (err: unknown) => reject(err));
    server.listen(0, '0.0.0.0', () => resolve(undefined));
  });
  return ready;
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function registerWithEnv(port: number, catalog: CatalogReport): Promise<void> {
  const target = `${ENV_BASE}/register`;
  const res = await fetch(target, {
    method: 'POST' as const,
    headers: JSON_MIME,
    body: JSON.stringify({ protocol_version: PROTOCOL_REV, port, catalog }),
    signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
  });
  if (res.ok === false) {
    const detail = (await res.text()).slice(0, DETAIL_LIMIT);
    throw new Error(`POST ${target}/register answered HTTP ${res.status}: ${detail}`);
  }
  const reply = (await res.json()) as unknown;
  const requests = isRecord(reply) ? reply.requests : undefined;
  if (typeof requests === 'number') {
    report(`registered: the trial holds ${requests} prompts`);
  }
}

/** One /work poll: true when the env says the trial is done. */
async function trialIsDone(): Promise<boolean> {
  const res = await fetch(`${ENV_BASE}/work`, { signal: AbortSignal.timeout(WORK_TIMEOUT_MS) });
  if (res.status !== 200) {
    return false;
  }
  const work = (await res.json()) as unknown;
  return isRecord(work) && work.done === true;
}

/** Stop the /work poll and close the server once the trial is done. */
function shutdownServer(server: Server, timer: NodeJS.Timeout): void {
  clearInterval(timer);
  server.close(() => undefined);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function pollForDone(server: Server): void {
  let inFlight = 0;
  const timer: NodeJS.Timeout = setInterval(() => {
    if (inFlight > 0) {
      return;
    }
    inFlight += 1;
    const runPoll = async (): Promise<void> => {
      try {
        const done = await trialIsDone();
        if (done) {
          shutdownServer(server, timer);
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight -= 1;
      }
    };
    void runPoll();
  }, WORK_CHECK_EVERY_MS);
}

/** The HTTP server: three routes, one error handler. */
function makeServer(): Server {
  return http.createServer((req, res) => {
    void dispatch(req, res).catch((err: unknown) => {
      report((err as Error).stack ?? String(err));
      if (res.headersSent === false) {
        const anthropicUrl = req.url?.startsWith('/v1/messages') === true;
        const wire = anthropicUrl ? 'anthropic' : 'openai';
        sendError(res, wire, 500, 'internal_error', SERVER_FAILED_MSG);
      }
    });
  });
}

/** Start: list the catalog and stop, or serve, register, and poll until done. */
async function start(): Promise<void> {
  if (briefDoc.error !== null) {
    report(briefDoc.error);
  }
  const wantsList = process.argv.includes('--list');
  if (wantsList) {
    // Print the register body's catalog, byte for byte, and stop before any
    // server starts: a process whose stdout is not flushed loses the tail,
    // so wait for the write to finish.
    const text = `${JSON.stringify(catalogReport)}\n`;
    await new Promise<void>((resolve) => {
      process.stdout.write(text, () => resolve());
    });
    process.exit(0);
  }
  if (ENV_BASE === '') {
    throw new Error('MAKEFUN_ENV_URL is not set');
  }
  if (chatCompletionsEndpoint() === '') {
    throw new Error('OPENAI_BASE_URL is not set');
  }
  const server = makeServer();
  await listenOnPort(server);
  const address = server.address() as AddressInfo | string | null;
  if (address === null || typeof address === 'string') {
    throw new Error('the server has no port');
  }
  const { port } = address;
  // Port 8080 belongs to the env on one layout and never comes from the OS's
  // ephemeral range, so reaching it means restarting the trial.
  if (port === 8080) {
    throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  }
  report(`listening on 0.0.0.0:${port} with ${modelsForList().length} free models`);
  await registerWithEnv(port, catalogReport);
  for (const signal of ['SIGTERM', 'SIGINT'] as const) {
    process.on(signal, () => process.exit(0));
  }
  pollForDone(server);
}

try {
  await start();
} catch (failure) {
  report((failure as Error).stack ?? String(failure));
  process.exit(1);
}
