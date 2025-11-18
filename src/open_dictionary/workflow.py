from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import logging
import sys
import time

from open_dictionary.db.access import DatabaseAccess
from open_dictionary.db.sqlite_manager import SQLiteManager
from typing import TYPE_CHECKING, Any
from datetime import datetime
import unicodedata
if TYPE_CHECKING:
    from open_dictionary.llm.define import Definition
from open_dictionary.wikitionary.downloader import DEFAULT_WIKTIONARY_URL, download_wiktionary_dump
from open_dictionary.wikitionary.extract import extract_wiktionary_dump
from open_dictionary.wikitionary.pre_process import _preprocess_payload, convert_to_toon
from pathlib import Path
import csv
import json
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class ProgressReporter:
    """Report progress of definition generation with statistics."""

    def __init__(
        self,
        *,
        min_time_step: float = 5.0,
        min_count_step: int = 10,
    ):
        self.min_time_step = max(min_time_step, 0.0)
        self.min_count_step = max(min_count_step, 1)
        self._last_report_time = time.monotonic()
        self._last_report_count = 0
        self._start_time = time.monotonic()

    def maybe_report(
        self,
        processed: int,
        failed: int,
        *,
        force: bool = False
    ) -> None:
        """Report progress if enough time/items have passed."""
        now = time.monotonic()
        count_increment = processed - self._last_report_count

        if not force:
            if (
                count_increment < self.min_count_step
                and (now - self._last_report_time) < self.min_time_step
            ):
                return

        elapsed = now - self._start_time
        total = processed + failed
        rate = processed / elapsed if elapsed > 0 else 0

        message = (
            f"Progress: {processed:,} processed | {failed:,} failed | "
            f"{total:,} total | {rate:.1f} items/sec"
        )
        logger.info(message)

        self._last_report_time = now
        self._last_report_count = processed

    def finalize(self, processed: int, failed: int) -> None:
        """Print final statistics."""
        elapsed = time.monotonic() - self._start_time
        total = processed + failed
        rate = processed / elapsed if elapsed > 0 else 0

        logger.info("=" * 60)
        logger.info(f"Processing complete!")
        logger.info(f"Total processed: {processed:,}")
        logger.info(f"Total failed: {failed:,}")
        logger.info(f"Total items: {total:,}")
        logger.info(f"Success rate: {(processed/total*100 if total > 0 else 0):.1f}%")
        logger.info(f"Total time: {elapsed:.1f} seconds")
        logger.info(f"Average rate: {rate:.1f} items/sec")
        logger.info("=" * 60)


def process_single_word(word_data: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Process a single word definition request.

    Args:
        word_data: Dictionary containing word data from PostgreSQL

    Returns:
        Tuple of (word, definition_dict) or None if processing failed
    """
    try:
        logger.debug(f"Processing word data keys: {list(word_data.keys())}")
        from open_dictionary.llm.define import define
        definition = define(word_data)
        result = (definition.word, definition.model_dump())
        logger.debug(f"Successfully processed word: {definition.word}")
        return result
    except Exception as e:
        logger.error(f"Failed to process word '{word_data.get('word', 'unknown')}': {e}", exc_info=True)
        return None


def run_parallel_definitions(
    table_name: str = "dictionary_en",
    batch_size: int = 50,
    max_workers: int = 50,
    sqlite_path: str = "data/dictionary.sqlite",
    limit: int | None = None,
):
    """Process dictionary entries in parallel and store in SQLite.

    This function reads from PostgreSQL, sends definition requests to LLM in parallel,
    and writes results to SQLite.

    Args:
        table_name: Name of the PostgreSQL table to read from
        batch_size: Number of rows to fetch from PostgreSQL per batch
        max_workers: Maximum number of parallel LLM requests
        sqlite_path: Path to SQLite database file
        limit: Optional limit on number of words to process
    """
    db_access = DatabaseAccess()
    sqlite_manager = SQLiteManager(sqlite_path)
    progress = ProgressReporter()

    logger.info(f"Starting parallel definition processing with {max_workers} workers")
    logger.info(f"Reading from PostgreSQL table: {table_name}")
    logger.info(f"Writing to SQLite: {sqlite_path}")
    if limit:
        logger.info(f"Processing limit: {limit:,} words")

    processed_count = 0
    failed_count = 0
    pending_batch = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Iterator to track all submitted futures
        future_to_word = {}

        # Iterate through PostgreSQL table
        row_iterator = db_access.iterate_table(
            table_name=table_name,
            batch_size=batch_size,
        )

        for row in row_iterator:
            # Check limit
            if limit and processed_count >= limit:
                break

            # Extract the data field if present (PostgreSQL stores JSON in 'data' column)
            word_data = row.get('data', row)
            word_key = word_data.get('word', 'unknown') if isinstance(word_data, dict) else 'unknown'

            # Submit word for processing
            future = executor.submit(process_single_word, word_data)
            future_to_word[future] = word_key

            # When we have max_workers futures pending, wait for some to complete
            if len(future_to_word) >= max_workers:
                # Wait for at least one to complete
                for future in as_completed(list(future_to_word.keys())):
                    # Process this completed future and break to continue submitting
                    if future not in future_to_word:
                        continue

                    word_key = future_to_word.pop(future)
                    result = future.result()

                    if result:
                        word, definition = result
                        pending_batch.append((word, definition))
                        processed_count += 1
                        logger.debug(f"Added '{word}' to pending batch (size: {len(pending_batch)})")

                        # Write batch when it reaches batch_size
                        if len(pending_batch) >= batch_size:
                            logger.debug(f"Writing batch of {len(pending_batch)} definitions to SQLite")
                            sqlite_manager.insert_definitions_batch(pending_batch)
                            logger.info(f"Wrote batch to SQLite. Total in DB: {sqlite_manager.count_definitions()}")
                            pending_batch = []

                        # Report progress
                        progress.maybe_report(processed_count, failed_count)
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to process: {word_key}")
                        progress.maybe_report(processed_count, failed_count)

                    # Break after processing one to continue submitting more work
                    break

        # Wait for remaining futures
        for future in as_completed(future_to_word.keys()):
            word_key = future_to_word[future]
            result = future.result()

            if result:
                word, definition = result
                pending_batch.append((word, definition))
                processed_count += 1
                progress.maybe_report(processed_count, failed_count)
            else:
                failed_count += 1
                logger.warning(f"Failed to process: {word_key}")
                progress.maybe_report(processed_count, failed_count)

        # Write any remaining definitions
        if pending_batch:
            logger.info(f"Writing final batch of {len(pending_batch)} definitions to SQLite")
            sqlite_manager.insert_definitions_batch(pending_batch)
            logger.info(f"Final batch written. Total in DB: {sqlite_manager.count_definitions()}")

    # Final statistics
    progress.finalize(processed_count, failed_count)
    final_count = sqlite_manager.count_definitions()
    logger.info(f"Total definitions in SQLite: {final_count:,}")

    if final_count != processed_count:
        logger.warning(f"Mismatch: processed {processed_count} but only {final_count} in database!")


def generate_json_from_csv(
    csv_path: str = "data/missing_words.csv",
    jsonl_path: str | None = None,
    output_dir: str = "data/words_json",
    overwrite: bool = False,
    use_toon: bool = False,
    max_workers: int = 20,
    auto_download: bool = True,
    source_url: str = DEFAULT_WIKTIONARY_URL,
    limit: int | None = None,
    fail_log: str | None = "data/words_json_failures.log",
    fail_jsonl: str | None = "data/words_json_failures.jsonl",
    timestamp_logs: bool = True,
    fallback_on_not_found: bool = False,
):
    words: list[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get("word") or "").strip()
            if w:
                words.append(w)

    if not words:
        return

    if limit is not None and limit > 0:
        words = words[:limit]

    workdir = Path("data")
    workdir.mkdir(parents=True, exist_ok=True)

    parsed = urllib.parse.urlparse(source_url)
    filename = Path(parsed.path or "raw-wiktextract-data.jsonl.gz").name
    gz_path = workdir / filename

    if jsonl_path is None:
        jsonl_file = workdir / "raw-wiktextract-data.jsonl"
    else:
        jsonl_file = Path(jsonl_path)

    if not jsonl_file.exists():
        if auto_download:
            download_wiktionary_dump(gz_path, url=source_url, overwrite=False)
            extract_wiktionary_dump(gz_path, jsonl_file, overwrite=True)
        else:
            raise FileNotFoundError(str(jsonl_file))

    targets: set[str] = set(words)
    payloads: dict[str, str] = {}

    def _normalize_variants(w: str) -> set[str]:
        s = w.strip()
        v: set[str] = set()
        v.add(s)
        v.add(s.lower())
        v.add(s.replace("-", " "))
        v.add(s.replace(" ", "-"))
        v.add(s.replace("'", ""))
        v.add(s.lower().replace("'", ""))
        nfkd = unicodedata.normalize("NFKD", s)
        ascii_only = "".join(ch for ch in nfkd if ord(ch) < 128)
        if ascii_only:
            v.add(ascii_only)
            v.add(ascii_only.lower())
        return v

    variant_map: dict[str, str] = {}
    for w in words:
        for v in _normalize_variants(w):
            variant_map[v] = w

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            w = obj.get("word")
            if not isinstance(w, str):
                continue
            mk = w if w in targets else variant_map.get(w)
            if mk is None:
                continue
            lang = obj.get("lang_code")
            if lang != "en":
                continue
            if use_toon:
                processed = _preprocess_payload(obj)
                payloads[mk] = convert_to_toon(processed)
            else:
                payloads[mk] = json.dumps(obj, ensure_ascii=False)
            if len(payloads) >= len(targets):
                break

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _task(word: str) -> dict[str, Any]:
        payload = payloads.get(word)
        if not payload:
            if fallback_on_not_found:
                stub = json.dumps({
                    "word": word,
                    "pos": "",
                    "forms": {},
                    "senses": [],
                    "etymology_text": "",
                }, ensure_ascii=False)
                payload = stub
            else:
                return {"ok": False, "word": word, "reason": "not_found_in_jsonl", "details": ""}
        from open_dictionary.llm.define import define
        try:
            d = define(payload)
            return {"ok": True, "word": d.word, "definition": d.model_dump(mode="json")}
        except Exception as exc:
            raw = getattr(exc, "llm_response", None)
            return {
                "ok": False,
                "word": word,
                "reason": "llm_error",
                "details": str(exc),
                "llm_response": raw if isinstance(raw, str) else None,
            }

    results_map: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_task, w): w for w in words}
        for future in as_completed(futures):
            w = futures[future]
            try:
                res = future.result()
            except Exception:
                res = {"ok": False, "word": w, "reason": "runtime_error", "details": ""}
            results_map[w] = res

    failed_words: list[str] = []
    failures: list[dict[str, Any]] = []
    for w in words:
        item = results_map.get(w)
        out_file = out_dir / f"{w}.json"
        if out_file.exists() and not overwrite:
            continue
        with open(out_file, "w", encoding="utf-8") as f:
            if item and item.get("ok"):
                definition = item.get("definition")
                json.dump(definition, f, ensure_ascii=False, indent=2)
            else:
                failed_words.append(w)
                err_reason = item.get("reason") if item else "unknown"
                err_obj: dict[str, Any] = {"word": w, "error": err_reason or "definition generation failed"}
                json.dump(err_obj, f, ensure_ascii=False, indent=2)
                fail_entry: dict[str, Any] = {"word": w, "reason": err_reason}
                if item and item.get("details"):
                    fail_entry["details"] = item.get("details")
                if item and item.get("llm_response"):
                    fail_entry["llm_response"] = item.get("llm_response")
                failures.append(fail_entry)

    if fail_log and failed_words:
        if timestamp_logs:
            p = Path(fail_log)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if p.suffix:
                p = p.with_name(f"{p.stem}_{ts}{p.suffix}")
            else:
                p = p.with_name(f"{p.name}_{ts}")
            fail_log = str(p)
        log_path = Path(fail_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            for w in failed_words:
                lf.write(f"{w}\n")
    if fail_jsonl and failures:
        if timestamp_logs:
            p = Path(fail_jsonl)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if p.suffix:
                p = p.with_name(f"{p.stem}_{ts}{p.suffix}")
            else:
                p = p.with_name(f"{p.name}_{ts}")
            fail_jsonl = str(p)
        jpath = Path(fail_jsonl)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        with open(jpath, "a", encoding="utf-8") as jf:
            for entry in failures:
                jf.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate dictionary definitions using LLM in parallel."
    )
    parser.add_argument(
        "--table",
        default="dictionary_en",
        help="PostgreSQL table to read dictionary entries from (default: dictionary_en).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of rows to fetch from PostgreSQL per batch (default: 50).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Maximum number of parallel LLM requests (default: 50).",
    )
    parser.add_argument(
        "--sqlite-path",
        default="data/dictionary.sqlite",
        help="Path to SQLite database file for storing definitions (default: data/dictionary.sqlite).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of words to process (for testing).",
    )

    args = parser.parse_args()

    run_parallel_definitions(
        table_name=args.table,
        batch_size=args.batch_size,
        max_workers=args.workers,
        sqlite_path=args.sqlite_path,
        limit=args.limit,
    )

