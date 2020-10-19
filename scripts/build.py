"""Build docker images for deepfigures.

See ``build.py --help`` for more information.
"""

import os
import logging
import json
import click

from deepfigures import settings
from scripts import execute

logger = logging.getLogger(__name__)


@click.command(
    context_settings={
        'help_option_names': ['-h', '--help']
    })
def build():
    """Build docker images for deepfigures."""
    for _, docker_img in settings.DEEPFIGURES_IMAGES.items():
        tag = docker_img['tag']
        dockerfile_path = docker_img['dockerfile_path']
        version = docker_img['version_prefix'] + settings.VERSION

        execute(
            'docker build'
            ' --tag {tag}:{version}'
            ' --cache-from {tag}:{version}'
            ' --build-arg BUILDKIT_INLINE_CACHE=1'
            ' --file {dockerfile_path} .'.format(
                tag=tag,
                version=version,
                dockerfile_path=dockerfile_path),
            logger)


@click.command(
    context_settings={
        'help_option_names': ['-h', '--help']
    })
@click.argument(
    'cpu_build_config_path',
    required=False,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        resolve_path=True))
@click.argument(
    'gpu_build_config_path',
    required=False,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        resolve_path=True))
def build_full(cpu_build_config_path, gpu_build_config_path):
    config_file_paths = []
    if cpu_build_config_path:
        config_file_paths.append(cpu_build_config_path)
    if gpu_build_config_path:
        config_file_paths.append(gpu_build_config_path)
    if not cpu_build_config_path and not gpu_build_config_path:
        config_file_paths = [settings.DOCKER_CPU_BUILD_CONFIG, settings.DOCKER_GPU_BUILD_CONFIG]

    for build_config_file in config_file_paths:
        build_config = json.load(open(build_config_file))
        config_dir = os.path.dirname(build_config_file)
        for stage_config in build_config["build_stages"]:
            execute(
                'docker build'
                ' --tag {user}/{repo}:{tag}'
                ' --cache-from {user}/{repo}:{tag}'
                ' --build-arg BUILDKIT_INLINE_CACHE=1'
                ' --file {docker_file} .'.format(
                    user=stage_config["user"],
                    repo=stage_config["repo"],
                    tag=stage_config["tag"],
                    docker_file=os.path.join(config_dir, stage_config["docker_file"])),
                logger)
            if stage_config["should_push"]:
                execute(
                    'docker push'
                    ' {user}/{repo}:{tag}'.format(user=stage_config["user"],
                                                  repo=stage_config["repo"],
                                                  tag=stage_config["tag"]),
                    logger
                )


if __name__ == '__main__':
    build()
