import { type KeyboardEvent, useMemo, useState } from "react";
import type { Lineage, Work } from "./api";

const WIDTH = 720;
const ROW = 30;
const TOP = 36;
const LEFT = 56;
const RIGHT = 24;
const UNKNOWN_GAP = 40;

interface Placed {
  work: Work;
  rank: number;
  x: number;
  y: number;
  r: number;
}

function ticks(first: number, last: number) {
  const step = Math.max(1, Math.ceil((last - first) / 6));
  const found = [];
  for (let year = Math.ceil(first / step) * step; year <= last; year += step) found.push(year);
  return found;
}

function layout(lineage: Lineage) {
  const works: Work[] = [lineage.target, ...lineage.originators];
  const years = works.flatMap((work) => (work.publication_year === null ? [] : [work.publication_year]));
  const first = Math.min(...years);
  const last = Math.max(...years);
  const undated = years.length < works.length;
  const start = LEFT + (undated ? UNKNOWN_GAP : 0);
  const x = (year: number | null) =>
    year === null ? LEFT : start + ((year - first) / (last - first || 1)) * (WIDTH - RIGHT - start);
  const top = Math.max(...lineage.originators.map((originator) => originator.score), 0) || 1;

  const placed = new Map<string, Placed>(
    works.map((work, rank) => [
      work.work_id,
      {
        work,
        rank,
        x: x(work.publication_year),
        y: TOP + rank * ROW,
        r: rank === 0 ? 9 : 4 + 6 * Math.sqrt(lineage.originators[rank - 1].score / top),
      },
    ]),
  );
  const axis = years.length ? ticks(first, last).map((year) => ({ label: String(year), x: x(year) })) : [];
  if (undated) axis.unshift({ label: "?", x: LEFT });
  return { placed, axis, height: TOP + works.length * ROW };
}

// Every shown work is reachable from the target, so the works leading to the
// focused one are exactly those on some path from the target to it.
function upstream(lineage: Lineage, focus: string) {
  const citedBy = new Map<string, string[]>();
  for (const link of lineage.links) citedBy.set(link.cited, [...(citedBy.get(link.cited) ?? []), link.citing]);
  const found = new Set([focus]);
  const pending = [focus];
  while (pending.length) {
    for (const citing of citedBy.get(pending.pop()!) ?? []) {
      if (!found.has(citing)) {
        found.add(citing);
        pending.push(citing);
      }
    }
  }
  return found;
}

// Ends at the rim of the cited circle, not its centre, so the arrowhead shows.
function curve(from: Placed, to: Placed) {
  if (Math.abs(from.x - to.x) < 1) {
    return `M${from.x},${from.y} L${to.x},${to.y + Math.sign(from.y - to.y) * (to.r + 2)}`;
  }
  const middle = (from.x + to.x) / 2;
  const end = to.x + Math.sign(from.x - to.x) * (to.r + 2);
  return `M${from.x},${from.y} C${middle},${from.y} ${middle},${to.y} ${end},${to.y}`;
}

function describe(node: Placed) {
  const { title, work_id, publication_year } = node.work;
  return `${node.rank === 0 ? "Target" : `#${node.rank}`}: ${title ?? work_id} (${publication_year ?? "year unknown"})`;
}

export default function GraphView({
  lineage,
  selected,
  onSelect,
}: {
  lineage: Lineage;
  selected: string | null;
  onSelect: (workId: string) => void;
}) {
  const { placed, axis, height } = useMemo(() => layout(lineage), [lineage]);
  const [hovered, setHovered] = useState<string | null>(null);
  const focus = hovered ?? selected;
  const lit = useMemo(() => (focus ? upstream(lineage, focus) : null), [lineage, focus]);
  const shown = focus ? placed.get(focus) : undefined;

  const press = (event: KeyboardEvent, workId: string) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    onSelect(workId);
  };

  return (
    <figure className="graph">
      <div className="scroll">
        <svg viewBox={`0 0 ${WIDTH} ${height}`} width={WIDTH} height={height} role="group" aria-label="Lineage graph">
          <defs>
            <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" className="arrowhead" />
            </marker>
          </defs>
          {axis.map((tick) => (
            <g key={tick.label} className="tick">
              <line x1={tick.x} x2={tick.x} y1={TOP - 16} y2={height - ROW / 2} />
              <text x={tick.x} y={TOP - 20}>
                {tick.label}
              </text>
            </g>
          ))}
          {lineage.links.map((link) => {
            const citing = placed.get(link.citing)!;
            const cited = placed.get(link.cited)!;
            const on = !lit || (lit.has(link.citing) && lit.has(link.cited));
            return (
              <path
                key={`${link.citing}-${link.cited}`}
                d={curve(citing, cited)}
                className={`link${link.direct ? "" : " indirect"}${on ? "" : " dim"}`}
                markerEnd="url(#arrow)"
              />
            );
          })}
          {[...placed.values()].map((node) => {
            const id = node.work.work_id;
            return (
              <g
                key={id}
                className={`node${node.rank === 0 ? " target" : ""}${lit && !lit.has(id) ? " dim" : ""}${
                  id === selected ? " selected" : ""
                }`}
                tabIndex={0}
                role="button"
                aria-pressed={id === selected}
                aria-label={describe(node)}
                onMouseEnter={() => setHovered(id)}
                onMouseLeave={() => setHovered(null)}
                onFocus={() => setHovered(id)}
                onBlur={() => setHovered(null)}
                onClick={() => onSelect(id)}
                onKeyDown={(event) => press(event, id)}
              >
                <title>{describe(node)}</title>
                <circle cx={node.x} cy={node.y} r={node.r} />
                <text x={node.x - node.r - 4} y={node.y} className="rank">
                  {node.rank === 0 ? "T" : node.rank}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <figcaption className="muted">
        {shown
          ? describe(shown)
          : "Older works to the left. Solid: cites it outright. Dashed: through works not shown. Hover or select a work to trace it back to the target."}
      </figcaption>
    </figure>
  );
}
