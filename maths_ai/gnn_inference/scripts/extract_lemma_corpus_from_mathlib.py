"""Extract a lemma corpus from the pinned Mathlib environment.

The companion script ``extract_lemma_corpus_from_hf.py`` builds a corpus from
the theorems the LeanDojo benchmark *proves*, which is the wrong population for
argument supervision: a tactic cites lemmas, and most cited lemmas are never
themselves a tactic-proved theorem.  Measured on the 5187 train rows that have a
version-3 syntax trace, that corpus covered 2308 of 9528 lemma citations
(24.22%), while the Mathlib environment covers all 9528.

This script therefore enumerates the environment itself.  ``env.catalog`` lists
every constant Lean knows about after importing Mathlib -- which transitively
includes Lean core and Batteries -- and ``env.inspect`` returns each constant's
type, used here as the lemma statement.

Usage::

    python -m maths_ai.gnn_inference.scripts.extract_lemma_corpus_from_mathlib \
        --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
        --pantograph-repl maths_ai/_support_files/sexpr_environment/Pantograph/.lake/build/bin/repl \
        --output-dir maths_ai/_support_files/artifacts/lemmas/v2/corpus

A measured full run wrote 318953 records in about 14 minutes for 143 MB, with no
inspect failures and no unsafe declarations skipped.  Run one instance at a time:
two writing the same output directory leave a partially written ``lemmas.jsonl``
that the audit reads as a small corpus rather than a broken one.

Pass ``--names-only`` to skip ``env.inspect`` entirely and write just the name
list that ``audit_argument_coverage --lemma-index`` consumes; that mode finishes
in seconds and is enough to label decoder targets, but carries no statements and
so cannot feed embeddings.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord, write_lemma_corpus


DEFAULT_OUTPUT_DIR = Path("maths_ai") / "_support_files" / "artifacts" / "lemmas" / "v2" / "corpus"
SAMPLE_SIZE = 50

# Pantograph prefixes every catalogued name with a single character naming the
# kind of constant it is.  Constructors are kept by default because tactics cite
# them: the audit classifies a CONSTRUCTOR position exactly like a GLOBAL one.
KIND_TAGS = {
    "t": "theorem",
    "d": "definition",
    "c": "constructor",
    "i": "inductive",
    "r": "recursor",
    "o": "other",
    "a": "axiom",
    "q": "quotient",
}
DEFAULT_KINDS = "".join(sorted(KIND_TAGS))

# Commands are written and replies read in blocks rather than one at a time, so
# a full run costs one round trip per block instead of per declaration, and
# never fills the pipe the way writing all 319000 commands up front would.
INSPECT_BATCH_SIZE = 512


@dataclass(frozen=True)
class ExtractionResult:
    catalog_size: int
    selected_names: int
    records: int
    unsafe_skipped: int
    inspect_failures: int
    sample_count: int
    names_only: bool


def _raise_stack_limit() -> None:
    """Raise this process's stack limit before Lean is spawned.

    Importing all of Mathlib overflows the default 8 MiB stack and Lean aborts
    with ``Stack overflow detected``.  The REPL inherits our limits, so lifting
    the soft limit to the hard one here removes the need for a wrapper shell
    that remembers to run ``ulimit -s``.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    if soft == hard:
        return
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    except (OSError, ValueError):
        pass


class PantographEnvironmentClient:
    """Synchronous Pantograph REPL client for environment-wide queries.

    The REPL answers exactly one JSON line per command and does not echo the
    name it was asked about, so replies are matched to requests by order.  That
    ordering is the only key available and every caller here depends on it.
    """

    def __init__(
        self,
        *,
        source_root: Path,
        pantograph_repl: Path,
        imports: str = "Mathlib",
        startup_timeout: int = 600,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.pantograph_repl = Path(pantograph_repl).resolve()
        self.imports = imports
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen[str] | None = None

    def start(self) -> "PantographEnvironmentClient":
        if self.proc is not None:
            return self
        if not self.pantograph_repl.exists():
            raise FileNotFoundError(f"Pantograph REPL does not exist: {self.pantograph_repl}")
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"Lean source root does not exist: {self.source_root}")
        _raise_stack_limit()
        self.proc = subprocess.Popen(
            ["lake", "env", "stdbuf", "-oL", str(self.pantograph_repl), self.imports],
            cwd=self.source_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert self.proc.stdout is not None
        ready = self.proc.stdout.readline()
        if ready.strip() != "ready.":
            detail = self._stderr_tail()
            self.close()
            raise RuntimeError(
                f"Pantograph emitted an invalid ready signal: {ready!r}. {detail}"
            )
        return self

    def _stderr_tail(self) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            return self.proc.stderr.read(4096).strip()
        except Exception:
            return ""

    def _write(self, command: str, payload: dict[str, object]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Pantograph client is not running.")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.proc.stdin.write(f"{command} {encoded}\n")

    def _read(self, command: str) -> dict[str, object]:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("Pantograph client is not running.")
        raw = self.proc.stdout.readline()
        if not raw:
            detail = self._stderr_tail()
            self.close()
            raise RuntimeError(f"Pantograph exited during '{command}'. {detail}")
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.close()
            raise RuntimeError(f"Pantograph returned invalid JSON for '{command}'.") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Pantograph returned a non-object for '{command}'.")
        return result

    def catalog(self) -> list[str]:
        """Return every constant name, still carrying its kind-tag prefix."""
        self._write("env.catalog", {})
        if self.proc is not None and self.proc.stdin is not None:
            self.proc.stdin.flush()
        result = self._read("env.catalog")
        if "error" in result:
            raise RuntimeError(f"Pantograph 'env.catalog' failed: {json.dumps(result)}")
        symbols = result.get("symbols", result)
        if not isinstance(symbols, list) or not all(isinstance(name, str) for name in symbols):
            raise RuntimeError("Pantograph 'env.catalog' did not return a list of names.")
        return symbols

    def inspect_many(self, names: Sequence[str]) -> Iterator[dict[str, object]]:
        """Yield one reply per requested name, in the order requested.

        A declaration the REPL rejects yields its error payload rather than
        raising, so one unreadable constant cannot abort a 319000-name run.
        """
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Pantograph client is not running.")
        for start in range(0, len(names), INSPECT_BATCH_SIZE):
            block = names[start : start + INSPECT_BATCH_SIZE]
            for name in block:
                self._write("env.inspect", {"name": name})
            self.proc.stdin.flush()
            for _ in block:
                yield self._read("env.inspect")

    def close(self) -> None:
        if self.proc is None:
            return
        proc, self.proc = self.proc, None
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


def _infer_namespace(full_name: str) -> str:
    """Infer namespace from a dotted fully-qualified name."""
    if "." in full_name:
        return full_name.rsplit(".", 1)[0]
    return ""


def _statement_of(payload: dict[str, object]) -> str:
    """Return the pretty-printed type carried by an ``env.inspect`` reply."""
    type_payload = payload.get("type")
    if not isinstance(type_payload, dict):
        return ""
    return str(type_payload.get("pp", "")).strip()


def select_catalog_names(catalog: Iterable[str], kinds: str) -> list[str]:
    """Strip kind tags and keep the requested kinds, preserving catalog order.

    A name is emitted once even when the environment lists it under more than
    one tag, because the corpus is keyed by name.
    """
    wanted = set(kinds)
    unknown = wanted - set(KIND_TAGS)
    if unknown:
        raise ValueError(f"Unknown kind tags requested: {''.join(sorted(unknown))}")
    selected: list[str] = []
    seen: set[str] = set()
    for tagged in catalog:
        if not tagged:
            continue
        tag, name = tagged[0], tagged[1:]
        if tag not in wanted or not name or name in seen:
            continue
        seen.add(name)
        selected.append(name)
    return selected


def extract_corpus(
    *,
    output_dir: Path,
    client_factory: Callable[[], PantographEnvironmentClient],
    kinds: str = DEFAULT_KINDS,
    names_only: bool = False,
    include_unsafe: bool = False,
    limit: int | None = None,
    force: bool = False,
    progress: Callable[[str], None] = print,
) -> ExtractionResult:
    """Enumerate the Lean environment and write a lemma corpus."""
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "lemmas.jsonl"
    sample_path = output_dir / "lemmas_sample.jsonl"
    names_path = output_dir / "lemma_names.json"
    manifest_path = output_dir / "manifest.json"
    failures_path = output_dir / "failures.jsonl"

    primary_path = names_path if names_only else corpus_path
    if primary_path.exists() and not force:
        raise FileExistsError(
            f"Output '{primary_path}' already exists. Use --force to overwrite."
        )

    client = client_factory().start()
    try:
        catalog = client.catalog()
        progress(f"  catalog returned {len(catalog)} constants")
        names = select_catalog_names(catalog, kinds)
        if limit is not None:
            names = names[:limit]
        progress(f"  selected {len(names)} names for kinds '{kinds}'")

        names_path.write_text(
            json.dumps(names, ensure_ascii=False), encoding="utf-8"
        )

        records = 0
        unsafe_skipped = 0
        sample_records: list[LemmaRecord] = []
        failures: list[dict[str, object]] = []

        if not names_only:
            # Records are streamed rather than collected and handed to
            # write_lemma_corpus, which takes a list: 319000 dataclasses held at
            # once costs hundreds of megabytes for no benefit.  The line format
            # is kept identical so both corpora load through the same reader.
            with corpus_path.open("w", encoding="utf-8") as handle:
                replies = client.inspect_many(names)
                for name, payload in zip(names, replies):
                    if "error" in payload:
                        failures.append(
                            {"name": name, "reason": "inspect_error", "payload": payload}
                        )
                        continue
                    if bool(payload.get("isUnsafe")) and not include_unsafe:
                        unsafe_skipped += 1
                        continue
                    statement = _statement_of(payload)
                    if not statement:
                        failures.append({"name": name, "reason": "empty_type"})
                        continue
                    record = LemmaRecord(
                        lemma_id=records,
                        name=name,
                        statement=statement,
                        namespace=_infer_namespace(name),
                        module=str(payload.get("module", "")),
                    )
                    handle.write(
                        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                    )
                    handle.write("\n")
                    if len(sample_records) < SAMPLE_SIZE:
                        sample_records.append(record)
                    records += 1
                    if records % 10000 == 0:
                        progress(f"  inspected {records} declarations...")
            write_lemma_corpus(sample_path, sample_records)
    finally:
        client.close()

    if failures:
        with failures_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
    else:
        # A clean rerun must not leave the previous run's failures behind,
        # where they would read as failures of this corpus.
        failures_path.unlink(missing_ok=True)

    manifest = {
        "source": "mathlib-environment",
        "kinds": kinds,
        "catalog_size": len(catalog),
        "selected_names": len(names),
        "records": records,
        "unsafe_skipped": unsafe_skipped,
        "inspect_failures": len(failures),
        "include_unsafe": include_unsafe,
        "limit": limit,
        "names_only": names_only,
        "statement_source": "env.inspect type.pp",
        "names_path": str(names_path),
        "corpus_path": None if names_only else str(corpus_path),
        "sample_path": None if names_only else str(sample_path),
        "sample_count": len(sample_records),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return ExtractionResult(
        catalog_size=len(catalog),
        selected_names=len(names),
        records=records,
        unsafe_skipped=unsafe_skipped,
        inspect_failures=len(failures),
        sample_count=len(sample_records),
        names_only=names_only,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a lemma corpus from the pinned Mathlib environment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for corpus artifacts",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Lean project the REPL is launched inside, normally the mathlib4 checkout",
    )
    parser.add_argument(
        "--pantograph-repl",
        type=Path,
        required=True,
        help="Path to the pinned Pantograph repl binary",
    )
    parser.add_argument(
        "--imports",
        type=str,
        default="Mathlib",
        help=(
            "Module the REPL imports before the environment is read. The default "
            "pulls in Lean core and Batteries transitively."
        ),
    )
    parser.add_argument(
        "--kinds",
        type=str,
        default=DEFAULT_KINDS,
        help=(
            "Kind tags to keep, as a string of single characters: "
            + ", ".join(f"{tag}={label}" for tag, label in sorted(KIND_TAGS.items()))
            + f" (default: {DEFAULT_KINDS})"
        ),
    )
    parser.add_argument(
        "--names-only",
        action="store_true",
        help=(
            "Write only lemma_names.json and skip env.inspect. Enough to label "
            "decoder targets, but carries no statements for embeddings."
        ),
    )
    parser.add_argument(
        "--include-unsafe",
        action="store_true",
        help="Keep declarations Lean marks unsafe (skipped by default)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on declarations inspected, for smoke tests",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing corpus artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    def client_factory() -> PantographEnvironmentClient:
        return PantographEnvironmentClient(
            source_root=args.source_root,
            pantograph_repl=args.pantograph_repl,
            imports=args.imports,
        )

    try:
        result = extract_corpus(
            output_dir=args.output_dir,
            client_factory=client_factory,
            kinds=args.kinds,
            names_only=args.names_only,
            include_unsafe=args.include_unsafe,
            limit=args.limit,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"\nExtracted lemma corpus:\n"
        f"  catalog constants = {result.catalog_size}\n"
        f"  selected names    = {result.selected_names}\n"
        f"  records written   = {result.records}\n"
        f"  unsafe skipped    = {result.unsafe_skipped}\n"
        f"  inspect failures  = {result.inspect_failures}\n"
        f"  sample entries    = {result.sample_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
