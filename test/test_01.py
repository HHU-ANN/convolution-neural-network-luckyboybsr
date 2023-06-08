# ！不要修改这里的内容
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from _01_Hello_World.Hello_World import main


def test():
    assert main() == "Hello, World!"


if __name__ == '__main__':
    main()
