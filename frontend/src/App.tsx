import { type FormEvent, useEffect, useRef, useState } from "react";
import {
  type Account,
  type FieldSummary,
  type Lineage,
  type Work,
  WORK_ID,
  fetchFields,
  fetchLineage,
  requestNarrative,
} from "./api";

type Request<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; data: T }
  | { status: "failed"; error: string };

const IDLE = { status: "idle" } as const;

// Only the latest call may settle, so a slow answer to an earlier question
// never overwrites the answer to the current one.
function useRequest<T>() {
  const [state, setState] = useState<Request<T>>(IDLE);
  const latest = useRef(0);
  const run = (call: () => Promise<T>) => {
    const ticket = ++latest.current;
    setState({ status: "loading" });
    call().then(
      (data) => ticket === latest.current && setState({ status: "done", data }),
      (error: Error) => ticket === latest.current && setState({ status: "failed", error: error.message }),
    );
  };
  const reset = () => {
    latest.current++;
    setState(IDLE);
  };
  return [state, run, reset] as const;
}

function WorkLink({ work }: { work: Pick<Work, "work_id" | "title"> }) {
  return (
    <a href={`https://openalex.org/${work.work_id}`} target="_blank" rel="noreferrer">
      {work.title ?? work.work_id}
    </a>
  );
}

function byline(work: Work) {
  const names = work.authors.length > 3 ? `${work.authors.slice(0, 3).join(", ")} et al.` : work.authors.join(", ");
  return [names, work.publication_year].filter(Boolean).join(", ");
}

function Failure({ request }: { request: Request<unknown> }) {
  return request.status === "failed" ? <p className="error">{request.error}</p> : null;
}

function LineageView({ lineage }: { lineage: Lineage }) {
  return (
    <section>
      <h2>
        Originators of <WorkLink work={lineage.target} />
      </h2>
      <p className="muted">{byline(lineage.target)}</p>
      {lineage.originators.length === 0 ? (
        <p>This paper has no ancestors in the field's snapshot.</p>
      ) : (
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Work</th>
                <th className="number">Score</th>
              </tr>
            </thead>
            <tbody>
              {lineage.originators.map((originator, index) => (
                <tr key={originator.work_id}>
                  <td>{index + 1}</td>
                  <td>
                    <WorkLink work={originator} />
                    <div className="muted">{byline(originator)}</div>
                  </td>
                  <td className="number">{originator.score.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <h3>Baseline overlap</h3>
      <p className="muted">Share of this top-k that each unlearned baseline also picked.</p>
      <ul className="baselines">
        {Object.entries(lineage.baselines).map(([name, overlap]) => (
          <li key={name}>
            {name}: {Math.round(overlap * 100)}%
          </li>
        ))}
      </ul>
    </section>
  );
}

function AccountView({ account }: { account: Account }) {
  return (
    <section>
      <h2>Why</h2>
      {account.partial && (
        <p className="warning">
          Partial account: some claims did not survive verification after {account.attempts} attempts.
        </p>
      )}
      <p>{account.summary}</p>
      <ol>
        {account.claims.map((claim) => (
          <li key={claim.work_id}>
            <WorkLink work={claim} />: {claim.contribution}
          </li>
        ))}
      </ol>
      {account.dropped.length > 0 && (
        <details>
          <summary>{account.dropped.length} claims dropped by verification</summary>
          <ul>
            {account.dropped.map((dropped, index) => (
              <li key={`${dropped.work_id}-${index}`}>
                {dropped.work_id}: {dropped.reason}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

export default function App() {
  const [fields, loadFields] = useRequest<FieldSummary[]>();
  const [lineage, loadLineage, resetLineage] = useRequest<Lineage>();
  const [account, narrate, resetAccount] = useRequest<Account>();
  const [field, setField] = useState("");
  const [workId, setWorkId] = useState("");

  useEffect(() => {
    loadFields(async () => {
      const found = await fetchFields();
      setField(found.find((candidate) => candidate.ready)?.name ?? "");
      return found;
    });
  }, []);

  const trimmed = workId.trim().toUpperCase();
  const validId = WORK_ID.test(trimmed);
  const selected = fields.status === "done" ? fields.data.find((candidate) => candidate.name === field) : undefined;
  const asked = lineage.status === "done" ? lineage.data : undefined;

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!validId || !field) return;
    resetAccount();
    loadLineage(() => fetchLineage(field, trimmed));
  };

  const changeField = (name: string) => {
    setField(name);
    resetLineage();
    resetAccount();
  };

  return (
    <main>
      <header>
        <h1>rootcite</h1>
        <p className="muted">Trace a paper back to its structural originators in a citation graph.</p>
      </header>

      <Failure request={fields} />
      <form onSubmit={submit}>
        <label>
          Field
          <select value={field} onChange={(event) => changeField(event.target.value)} disabled={fields.status !== "done"}>
            {fields.status === "done" &&
              fields.data.map((candidate) => (
                <option key={candidate.name} value={candidate.name} disabled={!candidate.ready}>
                  {candidate.display_name}
                  {candidate.ready ? "" : " (no snapshot)"}
                </option>
              ))}
          </select>
        </label>
        <label>
          OpenAlex work id
          <input
            value={workId}
            onChange={(event) => setWorkId(event.target.value)}
            placeholder="W2519887557"
            spellCheck={false}
            aria-invalid={workId !== "" && !validId}
          />
        </label>
        <button type="submit" disabled={!validId || !field || lineage.status === "loading"}>
          {lineage.status === "loading" ? "Ranking..." : "Trace"}
        </button>
      </form>
      {selected && <p className="muted">{selected.description}</p>}
      {fields.status === "done" && !fields.data.some((candidate) => candidate.ready) && (
        <p className="warning">No field has a snapshot yet. Run the pipeline first.</p>
      )}

      <Failure request={lineage} />
      {asked && (
        <>
          <LineageView lineage={asked} />
          {asked.originators.length > 0 && (
            <button
              onClick={() => narrate(() => requestNarrative(asked.field, asked.target.work_id))}
              disabled={account.status === "loading"}
            >
              {account.status === "loading" ? "Narrating..." : "Narrate (calls the model)"}
            </button>
          )}
        </>
      )}

      <Failure request={account} />
      {account.status === "done" && <AccountView account={account.data} />}
    </main>
  );
}
