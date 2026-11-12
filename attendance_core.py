# attendance_core.py
# Core functions for the school attendance study (one row = one student on one day).
# Paste this whole file into the first cell of a Google Colab notebook and run it.
# Students are only numbers (1, 2, 3 ...). No names or school details are used.

import re
import numpy as np
import pandas as pd

# ---- settings used in the study ---------------------------------------------
CHRONIC_BELOW = 90.0      # attendance % below this = chronically absent
RAINY_DAY_MM = 2.5        # rain of 2.5 mm or more = rainy day (IMD definition)
EARLY_DAYS = 22           # first four school weeks = 22 working days
DAYS_BEFORE_EXAM = 5      # window before an exam, in working days
DAYS_AFTER_EXAM = 3       # window after an exam, in working days
SPLIT_DATE = "2025-10-15" # first half / second half of the year

# Exam dates (first day, last day) for each class
EXAMS = {
    "Class VIII": {"Exam 1": ("2025-07-14", "2025-07-19"), "Exam 2": ("2025-10-06", "2025-10-13"),
                   "1 Sem": ("2025-12-18", "2025-12-24"), "Final": ("2026-03-27", "2026-04-07")},
    "Class IX":   {"Exam 1": ("2025-07-16", "2025-07-23"), "Exam 2": ("2025-09-22", "2025-09-27"),
                   "1 Sem": ("2025-12-18", "2025-12-24"), "Final": ("2026-03-23", "2026-04-01")},
}

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_WEEKDAY_OK = {"M": (0,), "T": (1, 3), "W": (2,), "F": (4,), "S": (5, 6)}


# ---- step 1: read the school's register into one tidy table ------------------
def _month_year(label):
    m = re.match(r"\s*([A-Za-z]+)\s*'?\s*(\d\d)", str(label))
    return 2000 + int(m.group(2)), _MONTHS[m.group(1)[:3].lower()]


def read_register_sheet(path, sheet, class_name, holidays):
    """Read one class sheet of the register (wide layout) into a long table."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    top, letters, days = raw.iloc[0], raw.iloc[1], raw.iloc[2]
    stud_rows = [i for i in range(3, len(raw)) if str(raw.iloc[i, 2]).startswith("Student")]
    rain_rows = [i for i in range(3, len(raw)) if str(raw.iloc[i, 2]).startswith("Rainfall")]
    rows, label = [], None
    for c in range(3, raw.shape[1]):
        if pd.notna(top[c]):
            label = top[c]
        day = days[c]
        if label is None or pd.isna(day) or not isinstance(day, (int, float, np.integer, np.floating)):
            continue
        if letters[c] not in _WEEKDAY_OK:
            continue
        year, month = _month_year(label)
        d = pd.Timestamp(year, month, int(day))
        if d.weekday() not in _WEEKDAY_OK[letters[c]]:
            raise ValueError(f"{sheet}: weekday letter {letters[c]} does not match {d.date()}")
        rain = np.nan
        if rain_rows:
            rain = pd.to_numeric(raw.iloc[rain_rows[0], c], errors="coerce")
        for r in stud_rows:
            status = raw.iloc[r, c]
            if status not in ("P", "A", "H"):
                continue
            num = int(re.search(r"(\d+)", str(raw.iloc[r, 2])).group(1))
            rows.append((class_name, num, d, status, rain if status != "H" else np.nan,
                         bool(status == "H" and d in holidays)))
    return pd.DataFrame(rows, columns=["cls", "student", "date", "status", "rain_mm", "listed_holiday"])


def load_register(path):
    """Read the school workbook (sheets 'Class VIII', 'Class IX', 'Holiday List')."""
    hol = pd.read_excel(path, sheet_name="Holiday List")
    holidays = set(pd.to_datetime(hol["Date"]))
    parts = [read_register_sheet(path, s, s, holidays) for s in ["Class VIII", "Class IX"]]
    return pd.concat(parts, ignore_index=True)


def load_tidy(path):
    """Read a tidy csv (cls, student, date, status, rain_mm, listed_holiday)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["listed_holiday"] = df["listed_holiday"].astype(bool)
    return df


# ---- step 2: add the columns used in the analysis -----------------------------
def prepare(df):
    """Keep working days (P or A) and add weekday, month, exam window and holiday flags."""
    df = df.copy()
    work_dates = {c: set(g.loc[g.status != "H", "date"]) for c, g in df.groupby("cls")}
    hol_dates = {c: set(g.loc[g.listed_holiday, "date"]) for c, g in df.groupby("cls")}

    # first working day after a break that contains a listed holiday
    after_hol = {}
    for c, wd in work_dates.items():
        after_hol[c] = set()
        for d in wd:
            k, seen = 1, False
            while k <= 10 and (d - pd.Timedelta(days=k)) not in wd:
                seen = seen or (d - pd.Timedelta(days=k)) in hol_dates[c]
                k += 1
            if seen:
                after_hol[c].add(d)

    # exam windows, counted in working days of that class
    window = {}
    for c, wd in work_dates.items():
        order = sorted(wd)
        window[c] = {}
        for name, (a, b) in EXAMS[c].items():
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            ex = [d for d in order if a <= d <= b]
            bef = [d for d in order if d < a][-DAYS_BEFORE_EXAM:]
            aft = [d for d in order if d > b][:DAYS_AFTER_EXAM]
            for d in bef:
                window[c].setdefault(d, ("before", name))
            for d in aft:
                window[c][d] = ("after", name)
            for d in ex:
                window[c][d] = ("exam", name)

    df = df[df.status != "H"].copy()
    df["absent"] = (df.status == "A").astype(int)
    df["weekday"] = df.date.dt.day_name()
    df["month"] = df.date.dt.to_period("M")
    df["after_holiday"] = [d in after_hol[c] for c, d in zip(df.cls, df.date)]
    w = [window[c].get(d, ("normal", "")) for c, d in zip(df.cls, df.date)]
    df["window"] = [x[0] for x in w]
    df["exam_name"] = [x[1] for x in w]
    df["rainy"] = df.rain_mm >= RAINY_DAY_MM
    df["day_no"] = df.groupby("cls").date.rank(method="dense").astype(int)
    return df.sort_values(["cls", "student", "date"]).reset_index(drop=True)


# ---- step 3: measures -------------------------------------------------------
def attendance_pct(df):
    """Attendance % = days present / working days x 100."""
    return 100.0 * (1.0 - df.absent.mean())


def student_table(df):
    """One row per student: working days, days absent, attendance %."""
    g = df.groupby(["cls", "student"]).absent.agg(["size", "sum"]).reset_index()
    g.columns = ["cls", "student", "working_days", "days_absent"]
    g["attendance_pct"] = 100.0 * (1.0 - g.days_absent / g.working_days)
    g["chronic"] = g.attendance_pct < CHRONIC_BELOW
    return g


def spell_lengths(absent_flags):
    """Lengths of runs of consecutive absences (in working days)."""
    runs, n = [], 0
    for a in absent_flags:
        if a:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def wilson_interval(k, n, z=1.96):
    """95% Wilson interval for a proportion k out of n (returned as fractions)."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def permutation_p(x, y, n_perm=10000, seed=42):
    """Two-sided permutation test for a difference in means (x versus y)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = x.mean() - y.mean()
    pool, nx = np.concatenate([x, y]), len(x)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:nx].mean() - pool[nx:].mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)



def paired_test(a, b, n_perm=10000, seed=42, n_boot=10000):
    """Paired comparison across students (a and b are lists, one value per student).
    Returns mean difference (a - b), 95% bootstrap range over students, and a
    sign-flip permutation p-value."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = d.mean()
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    cnt = sum(abs((d * rng.choice([-1, 1], len(d))).mean()) >= abs(obs) - 1e-12 for _ in range(n_perm))
    return obs, lo, hi, (cnt + 1) / (n_perm + 1)


def per_student_rate(df, mask):
    """Attendance % of each student on the days chosen by mask (students with no such day are dropped)."""
    g = df[mask].groupby("student").absent.mean()
    return 100.0 * (1.0 - g)

# ---- tests (run on made-up numbers, no school data needed) -------------------
if True:
    assert spell_lengths([0, 1, 1, 0, 1, 0, 0, 1, 1, 1]) == [2, 1, 3]
    toy = pd.DataFrame({"cls": "Class IX", "student": [1] * 4 + [2] * 4,
                        "absent": [0, 1, 0, 0, 0, 0, 0, 0]})
    st = student_table(toy)
    assert st.loc[0, "days_absent"] == 1 and abs(st.loc[0, "attendance_pct"] - 75.0) < 1e-9
    assert abs(attendance_pct(toy) - 87.5) < 1e-9
    lo, hi = wilson_interval(15, 18)
    assert 0.59 < lo < 0.62 and 0.93 < hi < 0.96
    obs, p = permutation_p([10, 11, 12, 13], [1, 2, 3, 4], n_perm=2000)
    assert obs == 9.0 and p < 0.05
    o, lo, hi, p = paired_test([5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6], n_perm=2000, n_boot=2000)
    assert o == 4.0 and lo == 4.0 and hi == 4.0 and p < 0.1
    print("Core functions loaded.")
    print("All checks passed.")
