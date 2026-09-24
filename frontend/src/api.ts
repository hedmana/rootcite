// Mirrors src/api/schemas.py. A change there is a change here.

export interface Work {
  work_id: string;
  title: string | null;
  publication_year: number | null;
  authors: string[];
}

export interface Originator extends Work {
  score: number;
}

export interface Link {
  citing: string;
  cited: string;
  direct: boolean;
}

export interface Lineage {
  field: string;
  target: Work;
  originators: Originator[];
  links: Link[];
  baselines: Record<string, number>;
}

export interface Cited {
  work_id: string;
  title: string | null;
  contribution: string;
}

export interface Dropped {
  work_id: string;
  reason: string;
}

export interface Account {
  field: string;
  target: Work;
  summary: string;
  claims: Cited[];
  dropped: Dropped[];
  attempts: number;
  partial: boolean;
}

export interface FieldSummary {
  name: string;
  display_name: string;
  description: string;
  ready: boolean;
}

export const WORK_ID = /^W\d+$/;

const BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {}

async function call<T>(path: string, method: "GET" | "POST" = "GET"): Promise<T> {
  let response: Response;
  try {
    response = await fetch(BASE + path, { method });
  } catch {
    throw new ApiError(`cannot reach the API at ${BASE}`);
  }
  if (response.ok) return response.json();
  const body = await response.json().catch(() => null);
  // FastAPI reports an HTTPException as a string and a validation failure as a list.
  const detail = body?.detail;
  const message = Array.isArray(detail)
    ? detail.map((problem: { msg: string }) => problem.msg).join("; ")
    : typeof detail === "string"
      ? detail
      : response.statusText;
  throw new ApiError(`${response.status}: ${message}`);
}

const segment = encodeURIComponent;

export const fetchFields = () => call<FieldSummary[]>("/fields");

export const fetchLineage = (field: string, workId: string) =>
  call<Lineage>(`/lineage/${segment(field)}/${segment(workId)}`);

export const requestNarrative = (field: string, workId: string) =>
  call<Account>(`/narrative/${segment(field)}/${segment(workId)}`, "POST");
