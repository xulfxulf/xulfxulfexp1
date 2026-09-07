import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train", "audit"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--output-dir", required=True)
        s.add_argument("--resume")
        if name == "audit":
            s.add_argument("--steps", type=int, default=4)
    s = sub.add_parser("verify-result")
    s.add_argument("--result", required=True)
    s = sub.add_parser("final-test")
    s.add_argument("--config", required=True)
    s.add_argument("--run-dir", required=True)
    s.add_argument("--selection-file", required=True)
    s.add_argument("--output-dir", required=True)
    args = p.parse_args()
    from research.runner import load_config, train, verify_result, final_test
    if args.command == "verify-result":
        verify_result(args.result)
    elif args.command == "final-test":
        final_test(load_config(args.config), args.run_dir, args.selection_file, args.output_dir)
    else:
        train(load_config(args.config), args.output_dir, " ".join(sys.argv), args.resume,
              args.steps if args.command == "audit" else None)


if __name__ == "__main__":
    main()
