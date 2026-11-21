"""Stages `publish`, `pages`, and `macos`: the only stages that change anything outside this machine.

Each runs only with `--confirm` (`cut.py publish --confirm`), after its checks pass, and `--dry-run` prints every
command and call it would make and changes nothing. Nothing here deletes or overwrites: a tag, a release, or a branch
file that exists already stops the stage, except the `macos` stage's one replacement of the release's `SHA256SUMS`, which
adds the macOS programs' lines and which the person approves first.

- `publish`: the public model repository gets the model's files and card as one commit and a tag; the GitHub release is
  created as a draft with its assets, which the person reviews and publishes.
- `macos`: after the release is public, the release-build workflow builds and tests the macOS programs from the tag, and
  they are attached under their asset names with a `SHA256SUMS` that names them too.
- `pages`: after the release is public, the site and browser page replace the `gh-pages` branch's files in one ordinary
  commit. The site links the macOS programs when `targets.toml` lists the platform, so the stage refuses while the release
  does not hold them, unless `--ahead-of-macos` is given.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import tomllib

from pathlib import Path

from . import receipts
from .commands import StageError, run
from .documents import unfinished
from .packaging import github_assets, sums_text
from .stages import Context, Stage, state


MACOS_ARTIFACTS = {"macos-arm64": ("litmus-macos-arm64", "litmus-app-macos-arm64"), "macos-x86_64": ("litmus-macos-x86_64", "litmus-app-macos-x86_64")}
BEFORE_PUBLISH = ("fetch", "record", "version", "score", "cost", "results", "bench", "browser", "docs", "build-windows", "build-linux", "verify", "reproduce", "assemble")


def targets(context: Context) -> dict:
    return tomllib.loads((context.repository / "release" / "targets.toml").read_text(encoding="utf-8"))


def git(context: Context, *arguments: str) -> str:
    return run(["git", *arguments], context.repository, capture=True).strip()


def gh_json(context: Context, *arguments: str) -> object:
    finished = subprocess.run(["gh", *arguments], cwd=context.repository, capture_output=True, text=True, encoding="utf-8", check=False)
    if finished.returncode != 0:
        return None
    return json.loads(finished.stdout) if finished.stdout.strip() else None


def gate(context: Context, stages: tuple[Stage, ...]) -> None:
    """Everything that must hold before anything is sent; raises with every failure."""
    found = []
    for stage in stages:
        if stage.name in BEFORE_PUBLISH and stage.required and state(context, stage)[0] != "done":
            found.append(f"stage {stage.name} is {state(context, stage)[0]}")
    if context.plan.repo == "local":
        found.append("the plan reads a local folder, not a Hub commit: a local trial is never published")
    found += unfinished(context)
    if git(context, "status", "--porcelain", "--", "."):
        found.append("the checkout has uncommitted changes: commit the release first")
    head = git(context, "rev-parse", "HEAD")
    for name in ("windows", "linux"):
        built = json.loads((context.work / "build" / name / "manifest.json").read_text(encoding="utf-8"))["source"]["commit"]
        if built != head:
            found.append(f"the {name} build was made at commit {built}, and the checkout is at {head}")
    remote = git(context, "ls-remote", "--heads", "origin")
    if head not in remote:
        found.append(f"commit {head} is not a branch head of origin: push the release commit (the repository may stay private until the release)")
    if git(context, "ls-remote", "--tags", "origin", f"refs/tags/{context.plan.tag}"):
        found.append(f"the tag {context.plan.tag} already exists on origin")
    if found:
        raise StageError("not ready to publish:\n" + "\n".join(found))


def say(context: Context, text: str) -> None:
    print(("[dry run] " if context.options.dry_run else "") + text, flush=True)


def act(context: Context, command: list[str], cwd: Path) -> None:
    say(context, "$ " + " ".join(command) + f"   (in {cwd})")
    if not context.options.dry_run:
        run(command, cwd)


def publish(context: Context, stages: tuple[Stage, ...]) -> None:
    gate(context, stages)
    settings = targets(context)
    hub_repo, github = settings["hub"]["model_repo"], settings["github"]["repo"]
    folder = context.work / "publish"
    head = git(context, "rev-parse", "HEAD")
    assets = sorted(path.name for path in (folder / "github").iterdir())
    say(context, f"Hub: upload {', '.join(sorted(path.name for path in (folder / 'hub').iterdir()))} to the model repository {hub_repo} as one commit \"Litmus {context.plan.version}\", then tag it {context.plan.tag}")
    if not context.options.dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        existing = [tag.name for tag in api.list_repo_refs(hub_repo, repo_type="model").tags]
        if context.plan.tag in existing:
            raise StageError(f"{hub_repo} already has the tag {context.plan.tag}; a published version is never overwritten")
        commit = api.upload_folder(folder_path=str(folder / "hub"), repo_id=hub_repo, repo_type="model", commit_message=f"Litmus {context.plan.version}")
        api.create_tag(hub_repo, tag=context.plan.tag, revision=commit.oid, repo_type="model")
        context.note("hub_commit", commit.oid)
    command = ["gh", "release", "create", context.plan.tag, "--repo", github, "--draft", "--target", head, "--title", f"Litmus {context.plan.version}",
               "--notes-file", str(folder / "github" / "NOTES.md"), *[str(folder / "github" / name) for name in assets if name != "NOTES.md"]]
    act(context, command, context.repository)
    say(context, f"then review the draft at https://github.com/{github}/releases and publish it (`gh release edit {context.plan.tag} --repo {github} --draft=false`); the `pages` stage waits for that")
    context.note("assets", assets)


def publish_outputs(context: Context) -> list[Path]:
    return [context.work / "publish" / "plan.json"]


def missing_macos_programs(settings: dict, release: object) -> list[str]:
    """The macOS programs the site links (`targets.toml` lists the platform) that the release does not hold."""
    if "macos" not in settings["programs"]["platforms"]:
        return []
    attached = {asset["name"] for asset in release["assets"]} if isinstance(release, dict) else set()
    return [name for pair in MACOS_ARTIFACTS.values() for name in pair if name not in attached]


def pages(context: Context, stages: tuple[Stage, ...]) -> None:
    settings = targets(context)
    github = settings["github"]
    release = gh_json(context, "release", "view", context.plan.tag, "--repo", github["repo"], "--json", "isDraft,tagName,assets")
    if not context.options.dry_run and (release is None or release["isDraft"]):  # type: ignore[index]
        raise StageError(f"the release {context.plan.tag} is not public yet: publish the draft, then run this stage")
    missing = missing_macos_programs(settings, release)
    if missing and context.options.ahead_of_macos:
        say(context, f"Pages: the site links {', '.join(missing)}, which the release does not hold yet; those links resolve once the `macos` stage has attached them")
    elif missing and context.options.dry_run:
        say(context, f"Pages: the site links {', '.join(missing)}, which the release does not hold yet: this stage will refuse until the `macos` stage has attached them, unless --ahead-of-macos is given")
    elif missing:
        raise StageError(f"the site links {', '.join(missing)}, which the release {context.plan.tag} does not hold: run the `macos` stage first, "
                         f"or pass --ahead-of-macos to publish the site now with links that resolve once the programs are attached")
    checkout = context.work / "pages_repo"
    url = f"https://github.com/{github['repo']}.git"
    say(context, f"Pages: replace the files of branch {github['pages_branch']} of {github['repo']} with {context.work / 'publish' / 'pages'} in one ordinary commit, no force")
    if context.options.dry_run:
        return
    shutil.rmtree(checkout, ignore_errors=True)
    heads = run(["git", "ls-remote", "--heads", url, github["pages_branch"]], context.repository, capture=True)
    if heads.strip():
        run(["git", "clone", "--branch", github["pages_branch"], "--single-branch", url, str(checkout)], context.repository)
        run(["git", "rm", "-r", "-q", "."], checkout)
    else:
        run(["git", "clone", url, str(checkout)], context.repository)
        run(["git", "switch", "--orphan", github["pages_branch"]], checkout)
    shutil.copytree(context.work / "publish" / "pages", checkout, dirs_exist_ok=True)
    run(["git", "add", "-A"], checkout)
    run(["git", "commit", "-m", f"Litmus {context.plan.version}: site and browser page"], checkout)
    run(["git", "push", "origin", github["pages_branch"]], checkout)


def pages_outputs(context: Context) -> list[Path]:
    return [context.work / "pages_repo" / "index.html"]


def workflow_commands(context: Context) -> list[list[str]]:
    github = targets(context)["github"]["repo"]
    base = f"https://github.com/{github}/releases/download/{context.plan.tag}/"
    return [["gh", "workflow", "run", "release-build.yml", "--repo", github, "--ref", context.plan.tag, "-f", f"model_url={base}"]]


def macos(context: Context, stages: tuple[Stage, ...]) -> None:
    github = targets(context)["github"]["repo"]
    folder = context.work / "macos"
    release = gh_json(context, "release", "view", context.plan.tag, "--repo", github, "--json", "isDraft,tagName")
    if not context.options.dry_run and (release is None or release["isDraft"]):  # type: ignore[index]
        raise StageError(f"the release {context.plan.tag} is not public yet: publish the draft, then run this stage (the workflow downloads the model from the release)")
    for command in workflow_commands(context):
        act(context, command, context.repository)
    say(context, f"wait for the run, download the artifacts litmus-macos-arm64 and litmus-macos-x86_64 into {folder}, check that each embeds this release's model, "
                 f"attach their programs as {', '.join(name for pair in MACOS_ARTIFACTS.values() for name in pair)}, and REPLACE the release's SHA256SUMS with one that also names them (the install scripts read it)")
    if context.options.dry_run:
        return
    time.sleep(20)
    runs = gh_json(context, "run", "list", "--repo", github, "--workflow", "release-build.yml", "--event", "workflow_dispatch", "--limit", "1", "--json", "databaseId")
    if not runs:
        raise StageError("the workflow run did not appear; look at the Actions tab of the repository and rerun this stage")
    identifier = str(runs[0]["databaseId"])  # type: ignore[index]
    act(context, ["gh", "run", "watch", identifier, "--repo", github, "--exit-status"], context.repository)
    shutil.rmtree(folder, ignore_errors=True)
    attached: dict[str, Path] = {}
    for artifact, (cli_name, app_name) in MACOS_ARTIFACTS.items():
        run(["gh", "run", "download", identifier, "--repo", github, "--name", f"litmus-{artifact}", "--dir", str(folder / artifact)], context.repository)
        manifest = json.loads((folder / artifact / "manifest.json").read_text(encoding="utf-8"))
        for name, entry in manifest["model"]["files"].items():
            if entry["sha256"] != receipts.file_digest(context.work / "model" / name):
                raise StageError(f"the {artifact} build embeds another {name} than this release's")
        attached[cli_name] = folder / artifact / "litmus"
        attached[app_name] = folder / artifact / "litmus-app"
    existing = github_assets(context)
    combined = sums_text({**existing, **attached})
    sums = folder / "SHA256SUMS"
    sums.write_text(combined, encoding="utf-8", newline="\n")
    named = {name: path for name, path in attached.items()}
    for name, path in named.items():
        shutil.copy2(path, folder / name)
    act(context, ["gh", "release", "upload", context.plan.tag, "--repo", github, "--clobber", str(sums), *[str(folder / name) for name in named]], context.repository)


def macos_outputs(context: Context) -> list[Path]:
    return [context.work / "macos" / "SHA256SUMS"]


def stages(upstream: tuple[Stage, ...]) -> tuple[Stage, ...]:
    """The three stages; `upstream` is every stage before `publish`, which its gate reads the state of."""
    def make(name: str, summary: str, function, outputs, needs: tuple[str, ...]) -> Stage:
        def run_with_upstream(context: Context) -> None:
            function(context, upstream)

        return Stage(name, needs, summary, run_with_upstream, outputs, gated=True, required=name == "publish")

    # The site links the macOS programs, so `macos` comes before `pages`, which checks that the release holds them.
    return (
        make("publish", "the Hub model repository and tag, and the GitHub release as a draft", publish, publish_outputs, ("assemble",)),
        make("macos", "the macOS programs from the release workflow, attached to the release", macos, macos_outputs, ("publish",)),
        make("pages", "the site and browser page on the gh-pages branch", pages, pages_outputs, ("publish",)),
    )
