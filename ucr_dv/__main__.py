"""python -m ucr_dv  -> the workbench GUI;  python -m ucr_dv run ... -> headless CLI."""
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('run', 'gate', 'eval', 'list', 'export'):
        from .cli import main as cli_main
        return cli_main(sys.argv[1:])
    from .app.main import main as gui_main
    return gui_main(sys.argv[1:])


if __name__ == '__main__':
    sys.exit(main())
