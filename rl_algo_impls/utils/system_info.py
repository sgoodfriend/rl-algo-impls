import logging
import re
import subprocess

import psutil
import torch


def log_cpu_info():
    commands = [
        "lscpu",  # Many Linux distributions
        "sysctl -n machdep.cpu",  # macOS
        "cat /proc/cpuinfo",  # Older/minimal Linux distributions
        "wmic cpu get name",  # Windows
    ]
    logger = logging.getLogger("CPU Info")
    for command in commands:
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
            )
            logger.info(result.stdout.strip())
            return
        except:
            continue
    else:
        logger.debug("Could not retrieve CPU information.")


def log_memory_info():
    logger = logging.getLogger("Memory Info")
    mem_info = psutil.virtual_memory()
    logger.info(f"Total memory: {mem_info.total / (1024**3):.2f} GB")
    logger.info(f"Available memory: {mem_info.available / (1024**3):.2f} GB")
    logger.info(f"Used memory: {mem_info.used / (1024**3):.2f} GB")
    logger.info(f"Memory percent used: {mem_info.percent}%")


def log_installed_libraries_info():
    logger = logging.getLogger("Libraries Info")
    logger.info(
        subprocess.check_output("python --version", shell=True).decode().strip()
    )
    logger.info(get_java_version())
    logger.info(f"torch: {torch.__version__}")
    logger.info(f"rl_algo_impls: {get_python_package_version('rl_algo_impls')}")


def get_python_package_version(package_name):
    result = subprocess.run(
        ["pip", "show", package_name], capture_output=True, text=True
    )
    version_line = re.search(r"^Version: (.*)$", result.stdout, re.MULTILINE)
    if version_line:
        return version_line.group(1)
    else:
        return None


def get_java_version() -> str:
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        return result.stderr.strip()
    except FileNotFoundError:
        return "Java is not installed."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log_cpu_info()
    log_memory_info()
    log_installed_libraries_info()
