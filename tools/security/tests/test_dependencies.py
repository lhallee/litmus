"""The Rust dependency record matches the lock files, and the license and network rules hold."""

import json
import check_dependencies


def test_record_matches_the_lock_files() -> None:
    record = json.loads(check_dependencies.RECORD.read_text(encoding="utf-8"))
    assert check_dependencies.problems_with(record) == []


def test_license_expressions() -> None:
    covered = check_dependencies.COVERED
    assert check_dependencies.satisfiable("MIT OR Apache-2.0", covered)
    assert check_dependencies.satisfiable("MIT/Apache-2.0", covered)
    assert check_dependencies.satisfiable("(MIT OR Apache-2.0) AND Unicode-3.0", covered)
    assert check_dependencies.satisfiable("Apache-2.0 OR GPL-2.0-only", covered)
    assert not check_dependencies.satisfiable("GPL-3.0-only", covered)
    assert not check_dependencies.satisfiable("(MIT OR Apache-2.0) AND GPL-2.0-only", covered)
    assert not check_dependencies.satisfiable("MPL-2.0", covered)
    assert check_dependencies.satisfiable("MPL-2.0", check_dependencies.NAMED)


def test_a_network_crate_and_an_unlisted_crate_are_problems() -> None:
    record = json.loads(check_dependencies.RECORD.read_text(encoding="utf-8"))
    record["engine/Cargo.lock"].append({"name": "reqwest", "version": "0.12.0", "license": "MIT OR Apache-2.0"})
    problems = check_dependencies.problems_with(record)
    assert any("reqwest 0.12.0 is a network crate" in problem for problem in problems)
    assert any("reqwest 0.12.0 is in dependencies.json but not locked" in problem for problem in problems)


def test_an_undocumented_copyleft_crate_is_a_problem() -> None:
    record = json.loads(check_dependencies.RECORD.read_text(encoding="utf-8"))
    for crate in record["app/Cargo.lock"]:
        if crate["name"] == "serde_json":
            crate["license"] = "GPL-3.0-only"
    problems = check_dependencies.problems_with(record)
    assert any("serde_json" in problem and "does not accept" in problem for problem in problems)
