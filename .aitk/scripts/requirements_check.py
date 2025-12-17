from pathlib import Path


def get_lines_from_file(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return (file_path, set(line.strip() for line in lines))


def req_is_subset(req1: tuple[Path, set[str]], req2: tuple[Path, set[str]], additional: set[str] = set()):
    if not additional.issubset(req1[1]):
        print(f"Additional is not a subset of '{req1[0].name}'")
        missing = additional - req1[1]
        print(f"Missing lines in '{req1[0].name}':")
        for req in missing:
            print(f"  {req}")
        raise SystemExit(1)
    req1 = (req1[0], req1[1] - additional)

    if req1[1].issubset(req2[1]):
        print(f"Requirement '{req1[0].name}' is subset of '{req2[0].name}'")
    else:
        print(f"Requirement '{req1[0].name}' is not a subset of '{req2[0].name}'")
        missing = req1[1] - req2[1]
        print(f"Missing lines in '{req2[0].name}':")
        for req in missing:
            print(f"  {req}")
        raise SystemExit(1)


def requirements_check():
    requirements_folder = Path(__file__).parent.parent / "requirements"
    WCR_lines = get_lines_from_file(requirements_folder / "requirements-WCR.txt")
    WCR_INIT_lines = get_lines_from_file(requirements_folder / "requirements-WCR_INIT.txt")
    QNN_LLM_lines = get_lines_from_file(requirements_folder / "requirements-QNN_LLM.txt")
    req_is_subset(WCR_INIT_lines, WCR_lines)
    req_is_subset(QNN_LLM_lines, WCR_lines)
    WCR_CUDA_lines = get_lines_from_file(requirements_folder / "requirements-WCR_CUDA.txt")
    req_is_subset(
        WCR_CUDA_lines,
        WCR_lines,
        additional={
            "--extra-index-url https://download.pytorch.org/whl/cu128",
            "# torch==2.7.0+cu128",
            "torch==2.7.0+cu128",
            "# torchvision==0.22.0+cu128",
            "torchvision==0.22.0+cu128",
        },
    )


if __name__ == "__main__":
    requirements_check()
