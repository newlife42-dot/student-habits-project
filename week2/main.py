import os
import sys

import numpy as np
import pandas as pd


# ==================================================
# 기능 1: 데이터 불러오기
# ==================================================
def load_data(file_path):
    """CSV 파일을 불러와 DataFrame으로 반환합니다."""

    # 파일이 실제로 존재하는지 확인
    if not os.path.exists(file_path):
        print(f"파일이 존재하지 않습니다: {file_path}")
        sys.exit()

    try:
        # 한글이 포함된 CSV 파일을 고려해 인코딩 지정
        df = pd.read_csv(file_path, encoding="utf-8-sig")

    except Exception as error:
        print(f"데이터를 불러오는 중 오류가 발생했습니다: {error}")
        sys.exit()

    rows, columns = df.shape
    print(f"데이터 로드 완료: {rows}행 × {columns}열")

    return df


# ==================================================
# 기능 2: 데이터 구조 확인
# ==================================================
def explore_structure(df):
    """행·열 수, 컬럼명, 자료형, 상위 5행을 출력합니다."""

    rows, columns = df.shape

    print("\n" + "=" * 60)
    print("1. 데이터 크기")
    print("=" * 60)
    print(f"전체 행 수: {rows}행")
    print(f"전체 열 수: {columns}열")

    print("\n" + "=" * 60)
    print("2. 컬럼 이름")
    print("=" * 60)

    for number, column in enumerate(df.columns, start=1):
        print(f"{number}. {column}")

    print("\n" + "=" * 60)
    print("3. 컬럼별 자료형")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("4. 상위 5개 데이터")
    print("=" * 60)
    print(df.head(5))


# ==================================================
# 기능 3: 기술통계 출력
# ==================================================
def show_statistics(df):
    """수치형 컬럼의 기술통계와 컬럼별 평균을 출력합니다."""

    # 기술통계 지표 설명
    # count: 결측치를 제외한 데이터 개수
    # mean: 평균값
    # std: 표준편차
    # min: 최솟값
    # 25%: 데이터를 작은 순서로 정렬했을 때 하위 25% 지점의 값
    # 50%: 중앙값
    # 75%: 데이터를 작은 순서로 정렬했을 때 하위 75% 지점의 값
    # max: 최댓값

    numeric_df = df.select_dtypes(include="number")

    print("\n" + "=" * 60)
    print("5. 수치형 컬럼 기술통계")
    print("=" * 60)

    if numeric_df.empty:
        print("수치형 컬럼이 없습니다.")
        return

    print(numeric_df.describe())

    print("\n" + "=" * 60)
    print("6. 수치형 컬럼별 평균")
    print("=" * 60)

    for column in numeric_df.columns:
        average = numeric_df[column].mean()
        print(f"{column}: {average:.2f}")


# ==================================================
# 기능 4: 결측치 현황 파악
# ==================================================
def check_missing(df):
    """결측치 수, 비율, 심각도를 출력하고 딕셔너리로 반환합니다."""

    missing_result = {}
    total_rows = len(df)

    print("\n" + "=" * 60)
    print("7. 결측치 현황")
    print("=" * 60)

    if total_rows == 0:
        print("데이터가 없습니다.")
        return missing_result

    # 각 컬럼의 결측치 개수 계산
    missing_counts = df.isnull().sum()

    for column in df.columns:
        missing_count = int(missing_counts[column])

        # 결측치가 1개 이상인 컬럼만 출력
        if missing_count == 0:
            continue

        missing_ratio = missing_count / total_rows * 100

        # 결측치 비율에 따른 심각도 판정
        if missing_ratio < 5:
            severity = "낮음"
        elif missing_ratio < 20:
            severity = "주의"
        else:
            severity = "높음"

        missing_result[column] = {
            "결측치 수": missing_count,
            "결측치 비율": round(missing_ratio, 2),
            "심각도": severity,
        }

        print(f"컬럼명: {column}")
        print(f"결측치 수: {missing_count}개")
        print(f"결측치 비율: {missing_ratio:.2f}%")
        print(f"심각도: {severity}")
        print("-" * 60)

    if not missing_result:
        print("결측치가 있는 컬럼이 없습니다.")

    return missing_result


# ==================================================
# 기능 5: NumPy로 직접 통계량 계산
# ==================================================
def numpy_stats(df, column="study_hours"):
    """NumPy로 통계량을 계산하고 pandas describe 결과와 비교합니다."""

    print("\n" + "=" * 60)
    print("8. NumPy 직접 통계량 계산")
    print("=" * 60)

    # 지정한 컬럼이 존재하는지 확인
    if column not in df.columns:
        print(f"컬럼이 존재하지 않습니다: {column}")
        return None

    # 결측치를 제거하고 NumPy 배열로 변환
    values = df[column].dropna().values

    if len(values) == 0:
        print(f"{column} 컬럼에 계산 가능한 값이 없습니다.")
        return None

    # NumPy로 통계량 계산
    numpy_mean = np.mean(values)

    # pandas describe()의 std와 비교하기 위해 ddof=1 사용
    numpy_std = np.std(values, ddof=1)

    numpy_median = np.median(values)
    numpy_min = np.min(values)
    numpy_max = np.max(values)

    print(f"분석 컬럼: {column}")
    print(f"평균: {numpy_mean:.2f}")
    print(f"표준편차: {numpy_std:.2f}")
    print(f"중앙값: {numpy_median:.2f}")
    print(f"최솟값: {numpy_min:.2f}")
    print(f"최댓값: {numpy_max:.2f}")

    # 조건 필터링: 하루 6시간 이상 공부하는 학생
    six_hours_or_more = values[values >= 6]
    print(f"6시간 이상 공부하는 학생 수: {len(six_hours_or_more)}명")

    # pandas describe() 결과 가져오기
    pandas_result = df[column].describe()

    print("\n" + "-" * 60)
    print("NumPy와 pandas describe() 결과 비교")
    print("-" * 60)

    print(
        f"평균 일치 여부: "
        f"{np.isclose(numpy_mean, pandas_result['mean'])}"
    )
    print(
        f"표준편차 일치 여부: "
        f"{np.isclose(numpy_std, pandas_result['std'])}"
    )
    print(
        f"중앙값 일치 여부: "
        f"{np.isclose(numpy_median, pandas_result['50%'])}"
    )
    print(
        f"최솟값 일치 여부: "
        f"{np.isclose(numpy_min, pandas_result['min'])}"
    )
    print(
        f"최댓값 일치 여부: "
        f"{np.isclose(numpy_max, pandas_result['max'])}"
    )

    result = {
        "평균": round(float(numpy_mean), 2),
        "표준편차": round(float(numpy_std), 2),
        "중앙값": round(float(numpy_median), 2),
        "최솟값": round(float(numpy_min), 2),
        "최댓값": round(float(numpy_max), 2),
        "6시간 이상 학생 수": len(six_hours_or_more),
    }

    return result


# ==================================================
# 기능 6: main() 함수로 전체 기능 연결
# ==================================================

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "student_habits.csv",
)


def main():
    """데이터 분석 기능을 순서대로 실행합니다."""

    # 기능 1: 데이터 불러오기
    df = load_data(DATA_PATH)

    # 기능 2: 데이터 구조 확인
    explore_structure(df)

    # 기능 3: 기술통계 출력
    show_statistics(df)

    # 기능 4: 결측치 현황 파악
    check_missing(df)

    # 기능 5: NumPy로 통계량 계산
    numpy_stats(df, "study_hours")


if __name__ == "__main__":
    main()

