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


if __name__ == "__main__":
    data_path = os.path.join("data", "student_habits.csv")
    df = load_data(data_path)
