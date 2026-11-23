"""Authenticate the completed paired DEV source review and obtain immutable TRAIN inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b11_hardnegative_pair as pair
import english_v6_b11_train_hardness as original

RUN = 45095073669
HEAD = "09f21a24ac84c380951ba51908e975f449a00bbb"
REPO = "repos/jiying2007/kws-data"
SOURCE = (14782806405, "47d3076b74ea5c6392e00b2bf110afcb91791e91a4a4b9e1fe78b0d2e2cf9bec")
RESULT = (14783505472, "5b8f2dfab1285eb9f544e538dd3508c2c1c9405df12c06f99ba03b61ca29ee8c")


def gh(endpoint):
    return subprocess.check_output(["gh", "api", REPO + endpoint])


def download(record, destination):
    aid, expected = record
    data = gh(f"/actions/artifacts/{aid}/zip")
    pair.need(hashlib.sha256(data).hexdigest() == expected, "immutable original Actions ZIP")
    destination.write_bytes(data)


def main(work):
    work.mkdir()
    run = json.loads(gh(f"/actions/runs/{RUN}"))
    pair.need(run["head_sha"] == HEAD and run["run_attempt"] == 1 and
              run["status"] == "completed" and run["conclusion"] == "success",
              "expected successful complete primary blind source QA")
    archives = json.loads(gh(f"/actions/runs/{RUN}/artifacts?per_page=100"))["artifacts"]
    review = [a for a in archives if a["name"] == f"english-v6-primary-dev-frozen-screen-{RUN}"]
    pair.need(len(review) == 1 and not review[0]["expired"] and
              review[0]["digest"].startswith("sha256:"), "unique immutable primary screen")
    item = review[0]
    digest = item["digest"].removeprefix("sha256:")
    download((item["id"], digest), work / "primary-review.zip")
    with ZipFile(work / "primary-review.zip") as ar:
        screen = ar.read("SCREEN.json")
    screen_sha = hashlib.sha256(screen).hexdigest()
    pair.reviewed_primary(screen, screen_sha, [])
    (work / "primary-screen.json").write_bytes(screen)
    with open(os.environ["GITHUB_ENV"], "a") as env:
        env.write(f"KWS_PRIMARY_SCREEN_SHA256={screen_sha}\nKWS_PRIMARY_REVIEW_ZIP_SHA256={digest}\n"
                  f"KWS_PRIMARY_REVIEW_ARTIFACT={item['id']}\n")
    print(f"Accepted independent primary screen {item['id']} SHA256 {digest}; member {screen_sha}")
    download(RESULT, work / "supplement-result.zip")
    download(SOURCE, work / "supplement-source.zip")
    train = work / "source"
    (train / "wav").mkdir(parents=True)
    with ZipFile(work / "supplement-source.zip") as ar:
        raw = ar.read("MANIFEST.json")
        (train / "MANIFEST.json").write_bytes(raw)
        rows = json.loads(raw)["rows"]
        pair.need(len(rows) == 24 and all(r["role"] == "train" for r in rows), "24 TRAIN originals")
        for row in rows:
            sid = row["id"]
            pair.need(sid.startswith("b11-train-") and Path(sid).name == sid, "registered TRAIN row")
            (train/"wav"/(sid+".wav")).write_bytes(ar.read("wav/"+sid+".wav"))
    pair.hard_rows(train, work / "supplement-result.zip", RESULT[1])
    dst = work / "original"
    dst.mkdir()
    for name, record in original.ARCHIVES.items():
        download(record, dst/(name+".zip"))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
