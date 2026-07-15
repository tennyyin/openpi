"""SageMaker launcher for openpi (JAX) fine-tuning on TRI fair-share Batch queues.

Adapted from open-world/sagemaker/launch_sm.py (same account/queue/spot/VPC handling).
Targets robotics-new (385697366450) + the cv-wfm p5en (H200) queue by default.

Usage:
    python sagemaker/launch_sm.py --config pi0_bike_rotor --name bike-rotor-pi0 \
        --queue cv-wfm --instance-count 1

Prereqs (run once, locally): convert the dataset, compute norm stats, then stage the
dataset + base checkpoints to S3 -- see sagemaker/README.md and sagemaker/stage_to_s3.sh.
"""

import argparse
from datetime import datetime
import inspect
import os
from pathlib import Path
import subprocess
import time

import boto3
from rich import print
from rich.table import Table
from sagemaker import Session as sm_Session
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

try:
    from sagemaker.aws_batch.training_queue import TrainingQueue as Queue
    _HAS_QUEUE = True
except ImportError:
    try:
        from sagemaker.batch_queueing.queue import Queue
        _HAS_QUEUE = True
    except ImportError:
        _HAS_QUEUE = False

NAME = "openpi"

INSTANCE_MAPPER = {
    "p4d": "ml.p4d.24xlarge", "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge", "p5en": "ml.p5en.48xlarge", "p6": "ml.p6-b200.48xlarge",
    "g6e": "ml.g6e.48xlarge", "g5": "ml.g5.48xlarge",
}

QUEUE_MAP = {
    ("cv", "cv-wfm", "cv-p5en", "cv-wfm-p5en"): ("fss-cv-wfm-p5en-48xlarge-us-west-2", "p5en"),
    ("cv-spot-p5", "cv-spot", "cv-p5en-spot"): ("fss-cv-wfm-spot-p5en-48xlarge-us-west-2", "p5en"),
    ("cv-spot-p6", "cv-p6-spot"): ("fss-cv-wfm-spot-p6-b200-48xlarge-us-west-2", "p6"),
}

_SUBNETS_US_WEST_2 = [
    "subnet-0610f766a4cd5cdae", "subnet-029adfb9e225d68f8", "subnet-01cc1bfeaf20155b5",
]

ACCOUNT_CONFIGS = {
    "robotics-new": dict(
        account_id="385697366450", profile="rob-sm",
        arn="arn:aws:iam::385697366450:role/Robotics-WFM-Sagemaker-role-us-west-2",
        s3_bucket="tri-ml-sandbox-16011-us-west-2-datasets",
        use_queue=True, max_run=5 * 24 * 60 * 60 - 360, volume_size=1000,
        # robotics-new runs jobs with NO VpcConfig (managed networking).
        subnets=None, security_group_ids=None,
    ),
    "robotics-old": dict(
        account_id="124224456861", profile="rob-s3",
        arn="arn:aws:iam::124224456861:role/SageMaker-SageMakerAllAccess-us-west-2",
        s3_bucket="tri-ml-sandbox-16011-us-west-2-datasets",
        use_queue=True, max_run=5 * 24 * 60 * 60 - 360, volume_size=1000,
        subnets=_SUBNETS_US_WEST_2, security_group_ids=["sg-00f130885f2e6ff96"],
    ),
}

TAGS = [
    {"Key": "tri.project", "Value": os.environ.get("TRI_PROJECT", "MM:PJ-0077")},
    {"Key": "tri.owner.email", "Value": os.environ.get("TRI_OWNER_EMAIL", "CHANGE.ME@tri.global")},
]

S3_ROOT = "s3://tri-ml-sandbox-16011-us-west-2-datasets/openpi"
# FastFile prefixes REQUIRE a trailing slash. dataset prefix contains <repo_id>/... so
# the mounted channel root works as HF_LEROBOT_HOME. base_ckpt holds {pi0_base,pi05_base}/.
S3_DATASET = f"{S3_ROOT}/datasets/"
S3_BASE_CKPT = f"{S3_ROOT}/base_ckpts/"


def resolve_wandb_api_key():
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if key:
        return key
    netrc = Path.home() / ".netrc"
    if netrc.exists():
        found = False
        for line in netrc.read_text().splitlines():
            toks = line.split()
            if "api.wandb.ai" in toks:
                found = True
            if found and "password" in toks:
                return toks[toks.index("password") + 1]
    raise SystemExit("WANDB_API_KEY empty and no wandb entry in ~/.netrc. Export it or `wandb login`.")


def run_command(command):
    print(f"[dim]=> {command}[/dim]")
    subprocess.run(command, shell=True, check=True)


def resolve_queue(queue_alias):
    for aliases, (queue_name, instance_type) in QUEUE_MAP.items():
        if queue_alias in aliases:
            return queue_name, instance_type
    all_aliases = [a for aliases in QUEUE_MAP for a in aliases]
    raise ValueError(f"Invalid queue: {queue_alias!r}. Valid: {all_aliases}")


def build_and_push_image(user, *, profile, region="us-west-2", build_type="full"):
    os.environ["AWS_PROFILE"] = profile
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    algorithm_name = f"{user}-{NAME}"
    dockerfile = Path(__file__).parent / "Dockerfile"
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"
    if build_type in (None, "None"):
        return fullname
    login = (f"aws ecr get-login-password --region {region} --profile {profile} "
             f"| docker login --username AWS --password-stdin")
    # The base image lives in robotics-old (124224456861); log in there to pull it.
    commands = [
        f"{login} 124224456861.dkr.ecr.{region}.amazonaws.com",
        f"{login} {account}.dkr.ecr.{region}.amazonaws.com",
        f"docker build -f {dockerfile} --build-arg AWS_REGION={region} -t {algorithm_name} .",
        f"docker tag {algorithm_name} {fullname}",
        (f"aws --region {region} --profile {profile} ecr describe-repositories --repository-names {algorithm_name} || "
         f"aws --region {region} --profile {profile} ecr create-repository --repository-name {algorithm_name}"),
    ]
    run_command("\n".join(f"{cmd} || exit 1" for cmd in commands))
    run_command(f"docker push {fullname}")
    time.sleep(5)
    return fullname


def parse_args():
    p = argparse.ArgumentParser(description="Launch openpi fine-tuning on SageMaker")
    p.add_argument("--account", default=None, choices=list(ACCOUNT_CONFIGS))
    p.add_argument("--build-type", default="full", help="full | None (skip build)")
    p.add_argument("--user", default=None, help="User name (or set SM_USER env var)")
    p.add_argument("--config", required=True, help="openpi config name, e.g. pi0_bike_rotor")
    p.add_argument("--exp-name", default=None, help="openpi exp_name (default: derived from --name)")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--profile", default=None)
    p.add_argument("--instance-count", default=1, type=int)
    p.add_argument("--queue", default="cv-wfm")
    p.add_argument("--priority", default=20, type=int)
    p.add_argument("--fss-identifier", default="default")
    p.add_argument("--name", default=None, help="Job name suffix")
    p.add_argument("--spot-instance", action="store_true")
    p.add_argument("--reserved", action="store_true")
    p.add_argument("--max-run-hours", default=None, type=float)
    p.add_argument("--s3-dataset", default=S3_DATASET, help="S3 prefix mounted as HF_LEROBOT_HOME")
    p.add_argument("--s3-base-ckpt", default=S3_BASE_CKPT, help="S3 prefix with {pi0_base,pi05_base}/params")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.user is None:
        args.user = os.environ.get("SM_USER")
        assert args.user, "Specify --user or set SM_USER env var"

    args.queue, args.instance_type = resolve_queue(args.queue)

    if "spot" in args.queue and not args.reserved:
        args.spot_instance = True
        if args.account is None:
            args.account = "robotics-old"
    if args.reserved:
        args.spot_instance = False
        if args.account is None:
            args.account = "robotics-old"
    if args.account is None:
        args.account = "robotics-new"

    acfg = ACCOUNT_CONFIGS[args.account]
    if args.profile is None:
        args.profile = acfg["profile"]

    os.environ["AWS_DEFAULT_REGION"] = args.region
    os.environ["SM_USE_RESERVED_CAPACITY"] = "0" if args.spot_instance else "1"

    image_uri = build_and_push_image(args.user, profile=args.profile, region=args.region, build_type=args.build_type)

    sagemaker_session = sm_Session(
        boto_session=boto3.session.Session(region_name=args.region, profile_name=args.profile),
        default_bucket=acfg["s3_bucket"],
    )

    base_job_name = f"{args.user}-{NAME}"
    if args.name is None:
        now = datetime.now()
        job_name = f"{base_job_name}-{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now.microsecond // 1000:03d}"
    else:
        job_name = f"{base_job_name}--{args.name}".replace("_", "-")
    exp_name = args.exp_name or (args.name or job_name).replace("-", "_")

    output_s3 = f"{S3_ROOT}/sagemaker/{args.user}/{job_name}"
    checkpoint_s3_uri = f"{S3_ROOT}/sagemaker/{args.user}/{job_name}/checkpoints"

    instance_type = INSTANCE_MAPPER[args.instance_type]
    max_run = int(args.max_run_hours * 3600) if args.max_run_hours else acfg["max_run"]
    max_wait = (max_run + 72 * 3600) if args.spot_instance else None

    hyperparameters = {"config": args.config, "exp_name": exp_name}

    environment = {
        "WANDB_API_KEY": resolve_wandb_api_key(),
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", "tri"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "openpi-bike-rotor"),
        "WANDB__SERVICE_WAIT": "300",
        "HF_HUB_OFFLINE": "1", "HF_HOME": "/tmp/hf_home",
        "TOKENIZERS_PARALLELISM": "false",
        "SAGEMAKER": "enabled",
        "SAGEMAKER_PROGRAM": "sagemaker/entrypoint.sh",
        "SM_JOB_NAME": job_name,
        "SM_USE_RESERVED_CAPACITY": "0" if args.spot_instance else "1",
    }
    for k in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "SM_HP_FSDP_DEVICES"):
        if os.environ.get(k):
            environment[k] = os.environ[k]

    fit_inputs = {
        "dataset": TrainingInput(s3_data=args.s3_dataset, input_mode="FastFile",
                                 s3_data_type="S3Prefix", distribution="FullyReplicated"),
        "base_ckpt": TrainingInput(s3_data=args.s3_base_ckpt, input_mode="File",
                                   s3_data_type="S3Prefix", distribution="FullyReplicated"),
    }

    table = Table(title=f"openpi SageMaker Job [{args.account}]", show_header=False,
                  title_style="bold cyan", border_style="dim")
    table.add_column("Key", style="bold"); table.add_column("Value")
    table.add_row("Account", f"{acfg['account_id']} ({args.account})")
    table.add_row("Image URI", image_uri)
    table.add_row("Job name", f"[bold green]{job_name}[/bold green]")
    table.add_row("Config / exp", f"{args.config}  /  {exp_name}")
    table.add_row("Instance", f"{args.instance_count}x {instance_type}")
    table.add_row("Queue", f"{args.queue}  (priority={args.priority})")
    table.add_row("Max runtime", f"{max_run}s (~{max_run / 3600:.1f}h)")
    table.add_row("Dataset S3", args.s3_dataset)
    table.add_row("Base ckpt S3", args.s3_base_ckpt)
    table.add_row("Checkpoint S3", checkpoint_s3_uri)
    print(table)

    disable_profiler = "p6-b200" in instance_type
    estimator = PyTorch(
        entry_point="sagemaker/entrypoint.sh",
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters=hyperparameters,
        role=acfg["arn"], image_uri=image_uri,
        instance_count=args.instance_count, instance_type=instance_type,
        train_use_spot_instances=args.spot_instance,
        output_path=output_s3, job_name=job_name,
        checkpoint_s3_uri=checkpoint_s3_uri, checkpoint_local_path="/opt/ml/checkpoints",
        code_location=output_s3, distribution=None, disable_profiler=disable_profiler,
        max_run=max_run, max_wait=max_wait, environment=environment,
        keep_alive_period_in_seconds=None if args.spot_instance else 300,
        volume_size=acfg["volume_size"], train_volume_size=acfg["volume_size"],
        enable_sagemaker_metrics=True, logs=True,
        tags=None if (acfg["use_queue"] and _HAS_QUEUE) else TAGS,
        subnets=acfg.get("subnets"), security_group_ids=acfg.get("security_group_ids"),
    )

    if args.dry_run:
        from sagemaker.estimator import _TrainingJob
        estimator.prepare_workflow_for_training(job_name)
        targs = _TrainingJob.get_train_args(estimator, fit_inputs, {})
        payload = estimator.sagemaker_session.get_train_request(**targs)
        print("[bold yellow]--- DRY RUN: payload Environment ---[/bold yellow]")
        print(payload.get("Environment") or {})
        print(f"[bold]Channels:[/bold] {list(fit_inputs.keys())}")
        return

    if acfg["use_queue"] and _HAS_QUEUE:
        queue = Queue(args.queue)
        print(f"[bold green]Submitting to queue: {getattr(queue, 'queue_name', args.queue)}[/bold green]")
        map_kwargs = dict(
            inputs=[fit_inputs], job_names=[job_name], priority=args.priority,
            share_identifier=args.fss_identifier,
            timeout={"attemptDurationSeconds": max_wait or max_run},
            tags={t["Key"]: t["Value"] for t in TAGS},
        )
        if "training_job" in inspect.signature(queue.map).parameters:
            queue.map(training_job=estimator, **map_kwargs)
        else:
            queue.map(estimator=estimator, **map_kwargs)
    else:
        print(f"[bold green]Submitting training job: {job_name}[/bold green]")
        estimator.fit(inputs=fit_inputs, wait=False, job_name=job_name)


if __name__ == "__main__":
    main()
