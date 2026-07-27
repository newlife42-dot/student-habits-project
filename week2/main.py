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

# ==================================================
# Week2
# ==================================================
def handle_missing(df):
    """결측치를 중앙값으로 대체하고 처리 결과를 반환합니다."""

    # 원본 DataFrame을 보존하기 위해 복사
    processed_df = df.copy()

    # 결측치를 처리할 컬럼 목록
    target_columns = [
        "sleep_hours",
        "phone_hours",
        "exercise_hours",
    ]

    print("\n" + "=" * 60)
    print("1. 결측치 처리")
    print("=" * 60)

    for column in target_columns:
        if column not in processed_df.columns:
            print(f"{column}: 컬럼이 존재하지 않습니다.")
            continue

        missing_count = processed_df[column].isnull().sum()
        median_value = processed_df[column].median()

        # 결측치를 해당 컬럼의 중앙값으로 대체
        processed_df[column] = processed_df[column].fillna(median_value)

        print(f"{column}")
        print(f"- 처리 전 결측치: {missing_count}개")
        print(f"- 대체 중앙값: {median_value:.2f}")
        print(f"- 처리 후 결측치: {processed_df[column].isnull().sum()}개")

    # 전체 DataFrame에 남아 있는 결측치 개수
    remaining_missing = processed_df.isnull().sum().sum()

    print("-" * 60)
    print(f"결측치 처리 후 남은 결측치: {remaining_missing}개")

    return processed_df

def handle_outliers(df):
    """IQR 방식으로 이상치를 탐지하고 경계값 안으로 클리핑합니다."""

    processed_df = df.copy()

    target_columns = [
        "sleep_hours",
        "study_hours",
        "phone_hours",
        "exercise_hours",
    ]

    print("\n" + "=" * 60)
    print("2. 이상치 처리")
    print("=" * 60)

    for column in target_columns:
        if column not in processed_df.columns:
            print(f"{column}: 컬럼이 존재하지 않습니다.")
            continue

        # 1사분위수와 3사분위수 계산
        q1 = processed_df[column].quantile(0.25)
        q3 = processed_df[column].quantile(0.75)

        # 사분위 범위 계산
        iqr = q3 - q1

        # 이상치 경계 계산
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 처리 전 이상치 개수
        outlier_count = (
            (processed_df[column] < lower_bound)
            | (processed_df[column] > upper_bound)
        ).sum()

        before_min = processed_df[column].min()
        before_max = processed_df[column].max()

        # 범위를 벗어난 값을 하한·상한값으로 대체
        processed_df[column] = processed_df[column].clip(
            lower=lower_bound,
            upper=upper_bound,
        )

        after_min = processed_df[column].min()
        after_max = processed_df[column].max()

        print(f"컬럼명: {column}")
        print(f"- Q1: {q1:.2f}")
        print(f"- Q3: {q3:.2f}")
        print(f"- IQR: {iqr:.2f}")
        print(f"- 하한: {lower_bound:.2f}")
        print(f"- 상한: {upper_bound:.2f}")
        print(f"- 처리된 이상치: {outlier_count}개")
        print(f"- 처리 전 범위: {before_min:.2f} ~ {before_max:.2f}")
        print(f"- 처리 후 범위: {after_min:.2f} ~ {after_max:.2f}")
        print("-" * 60)

    print("이상치 처리(클리핑) 완료")

    return processed_df

def convert_types(df):
    """gender 컬럼을 0과 1로 인코딩해 gender_code 컬럼을 추가합니다."""

    processed_df = df.copy()

    print("\n" + "=" * 60)
    print("3. 자료형 인코딩")
    print("=" * 60)

    if "gender" not in processed_df.columns:
        print("gender 컬럼이 존재하지 않습니다.")
        return processed_df

    print(f"변환 전 gender 고유값: {processed_df['gender'].unique()}")

    # gender가 '여'일 때 1, 그 외에는 0
    processed_df["gender_code"] = (
        processed_df["gender"] == "여"
    ).astype(int)

    print(
        "gender_code 고유값:",
        processed_df["gender_code"].unique(),
    )

    print("자료형 인코딩 완료")

    return processed_df

def add_features(df):
    """기존 컬럼을 활용해 3개의 파생변수를 생성합니다."""

    processed_df = df.copy()

    print("\n" + "=" * 60)
    print("4. 파생변수 생성")
    print("=" * 60)

    required_columns = [
        "study_hours",
        "exercise_hours",
        "sleep_hours",
        "phone_hours",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in processed_df.columns
    ]

    if missing_columns:
        print(
            "필요한 컬럼이 존재하지 않습니다:",
            missing_columns,
        )
        return processed_df

    # 공부 시간 + 운동 시간
    processed_df["productive_hours"] = (
        processed_df["study_hours"]
        + processed_df["exercise_hours"]
    )

    # 수면 시간이 7시간 이상이면 1, 아니면 0
    processed_df["sleep_sufficient"] = (
        processed_df["sleep_hours"] >= 7
    ).astype(int)

    # 스마트폰 사용 시간이 4시간을 초과하면 1, 아니면 0
    processed_df["phone_overuse"] = (
        processed_df["phone_hours"] > 4
    ).astype(int)

    print("생성된 파생변수:")
    print("- productive_hours")
    print("- sleep_sufficient")
    print("- phone_overuse")

    print("\n파생변수 상위 5행:")
    print(
        processed_df[
            [
                "productive_hours",
                "sleep_sufficient",
                "phone_overuse",
            ]
        ].head()
    )

    print("파생변수 생성 완료")

    return processed_df

# ==================================================
# 파일 경로 상수
# ==================================================
PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "student_habits.csv",
)

OUTPUT_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "student_habits_clean.csv",
)


# ==================================================
# 기능 5: 전체 연결 및 저장
# ==================================================
def main():
    """전처리 기능을 순서대로 실행하고 결과를 CSV로 저장합니다."""

    # 원본 데이터 불러오기
    df = load_data(INPUT_PATH)

    # 기능 1: 결측치 처리
    df = handle_missing(df)

    # 기능 2: 이상치 처리
    df = handle_outliers(df)

    # 기능 3: 자료형 인코딩
    df = convert_types(df)

    # 기능 4: 파생변수 생성
    df = add_features(df)

    # 전처리 결과 저장
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("전처리 데이터 저장 완료")
    print("=" * 60)
    print(f"저장 경로: {OUTPUT_PATH}")
    print(f"최종 데이터 크기: {df.shape[0]}행 × {df.shape[1]}열")


if __name__ == "__main__":
    main()
