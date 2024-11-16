import NorthRNP.levelRNP as levelRNP
import SouthRNP.SlevelRNP as SlevelRNP
import NorthRNP.verticalRNP as verticalRNP
import SouthRNP.SverticalRNP as SverticalRNP


def main():
    """
    主函数，依次运行 levelRNP, SlevelRNP, verticalRNP, SverticalRNP 四个模块
    """
    print("Running levelRNP...")
    levelRNP.main()  # 假设 levelRNP 文件中有 main() 函数作为入口

    print("Running SlevelRNP...")
    SlevelRNP.main()  # 假设 SlevelRNP 文件中有 main() 函数作为入口

    print("Running verticalRNP...")
    verticalRNP.main()  # 假设 verticalRNP 文件中有 main() 函数作为入口

    print("Running SverticalRNP...")
    SverticalRNP.main()  # 假设 SverticalRNP 文件中有 main() 函数作为入口

    print("All modules have been executed.")


if __name__ == "__main__":
    main()
