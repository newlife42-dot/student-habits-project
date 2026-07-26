import os
import sys

import pandas as pd


def load_data(file_path):
    """CSV 파일을 불러와 DataFrame으로 반환합니다."""

    if not os.path.exists(file_path):
        print(f"파일이 존재하지 않습니다: {file_path}")
        sys.exit()

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception as error:
        print(f"데이터를 불러오는 중 오류가 발생했습니다: {error}")
        sys.exit()

    rows, columns = df.shape
    print(f"데이터 로드 완료: {rows}행 × {columns}열")

    return df


def explore_structure(df):
    """DataFrame의 행·열 수, 컬럼명, 자료형, 상위 5행을 출력합니다."""

    rows, columns = df.shape

    print("\n" + "=" * 50)
    print("1. 데이터 크기")
    print("=" * 50)
    print(f"전체 행 수: {rows}행")
    print(f"전체 열 수: {columns}열")

    print("\n" + "=" * 50)
    print("2. 컬럼 이름")
    print("=" * 50)

    for number, column in enumerate(df.columns, start=1):
        print(f"{number}. {column}")

    print("\n" + "=" * 50)
    print("3. 컬럼별 자료형")
    print("=" * 50)
    print(df.dtypes)

    print("\n" + "=" * 50)
    print("4. 상위 5개 데이터")
    print("=" * 50)
    print(df.head(5))


if __name__ == "__main__":
    data_path = os.path.join("data", "student_habits.csv")

    df = load_data(data_path)
    explore_structure(df)
