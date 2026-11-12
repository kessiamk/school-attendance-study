# step2_analysis.py
# Run this AFTER attendance_core.py, in the same Colab notebook.
# It reads the register, prints the tables used in the write-up and saves two charts.
# It reads attendance_register.csv (the school register, one row per student per day;
# upload this file to Colab first). Students are numbers only. The file name can also be
# a school workbook (.xlsx) in the same layout as the original register.

import json
import matplotlib.pyplot as plt

DATA_PATH = "attendance_register.csv"

raw = load_tidy(DATA_PATH) if DATA_PATH.endswith(".csv") else load_register(DATA_PATH)
df = prepare(raw)
st = student_table(df)
CLASSES = ["Class VIII", "Class IX"]
COL = {"Class VIII": "#d62728", "Class IX": "#1f77b4"}   # red = VIII, blue = IX
R = {}                                                   # results kept for the write-up


def pct(mask_df):
    return round(attendance_pct(mask_df), 2)


# ---- Table 1: the two classes ---------------------------------------------------
t1 = []
for c in CLASSES:
    g, s = df[df.cls == c], st[st.cls == c]
    t1.append(dict(cls=c, students=len(s), working_days=int(g.date.nunique()), student_days=len(g),
                   absences=int(g.absent.sum()), attendance=pct(g), mean_days_absent=round(s.days_absent.mean(), 1),
                   chronic=int(s.chronic.sum()), lowest=round(s.attendance_pct.min(), 2),
                   highest=round(s.attendance_pct.max(), 2)))
R["table1"] = t1
print(pd.DataFrame(t1).to_string(index=False))

# H1: is the difference between the classes bigger than chance?
diff, p = permutation_p(st[st.cls == "Class IX"].attendance_pct, st[st.cls == "Class VIII"].attendance_pct)
R["H1"] = dict(diff=round(diff, 2), p=p)
print(f"\nH1: Class IX minus Class VIII = {diff:.2f} percentage points (permutation p = {p:.4f})")

# ---- Month by month (Figure 1) -------------------------------------------------
months = df.groupby(["cls", "month"]).absent.apply(lambda s: round(100 * (1 - s.mean()), 2)).unstack(0)
R["months"] = {c: {str(k): v for k, v in months[c].items()} for c in CLASSES}
print("\nAttendance by month (%):\n", months.to_string())

fig, ax = plt.subplots(figsize=(12, 6.75))
x = range(len(months))
for c, mk in zip(CLASSES, ["o", "s"]):
    ax.plot(x, months[c], marker=mk, color=COL[c], lw=2.2, label=f"{c}: attendance in each month")
ax.axhline(90, color="grey", ls=":", lw=1.5, label="90% line (below this = chronically absent)")
ax.axhline(75, color="red", ls="--", lw=1.2, label="75% line (needed for public examinations)")
ax.set_xticks(list(x))
ax.set_xticklabels([p.strftime("%b %y") for p in months.index], rotation=45)
ax.set_ylabel("Attendance (% of working days)")
ax.set_ylim(60, 100)
ax.set_title("Attendance month by month, 2025-26")
ax.legend(loc="lower left", title="Each line follows one class")
fig.tight_layout()
fig.savefig("fig1_months.png", dpi=100)
plt.close(fig)

# ---- Weekday (H2) --------------------------------------------------------------
wk = df.groupby(["cls", "weekday"]).absent.apply(lambda s: round(100 * (1 - s.mean()), 2)).unstack(0)
R["weekday"] = {c: wk[c].to_dict() for c in CLASSES}
print("\nAttendance by weekday (%):\n", wk.to_string())
R["H2"] = {}
for c in CLASSES:
    g = df[df.cls == c]
    side = g.weekday.isin(["Monday", "Friday", "Saturday"])
    a, b = per_student_rate(g, side), per_student_rate(g, ~side)
    o, lo, hi, p = paired_test(a, b)
    R["H2"][c] = dict(diff=round(o, 2), lo=round(lo, 2), hi=round(hi, 2), p=p)
    print(f"H2 {c}: Mon/Fri/Sat minus other days = {o:.2f} points (95% range {lo:.2f} to {hi:.2f}, p = {p:.3f})")

# ---- Examination windows (H4, H5, Figure 2) ---------------------------------------
order = ["before", "exam", "after", "normal"]
labels = {"before": "5 days before\nan exam", "exam": "exam days",
          "after": "3 days after\nan exam", "normal": "normal days"}
win = df.groupby(["cls", "window"]).absent.apply(lambda s: round(100 * (1 - s.mean()), 2)).unstack(0).loc[order]
R["windows"] = {c: win[c].to_dict() for c in CLASSES}
print("\nAttendance around examinations (%):\n", win.to_string())

fig, ax = plt.subplots(figsize=(12, 6.75))
w = 0.38
for i, c in enumerate(CLASSES):
    bars = ax.bar([k + (i - 0.5) * w for k in range(4)], win[c], w, color=COL[c],
                  label=f"{c}: attendance on these days")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.1f}",
                ha="center", fontsize=10)
ax.set_xticks(range(4))
ax.set_xticklabels([labels[k] for k in order])
ax.set_ylim(60, 110)
ax.set_ylabel("Attendance (% of working days)")
ax.set_title("Attendance around examinations")
ax.legend(loc="upper center", ncol=2, title="Each bar = average attendance on those days")
fig.tight_layout()
fig.savefig("fig2_exams.png", dpi=100)
plt.close(fig)

R["H4"], R["H5"] = {}, {}
for c in CLASSES:
    g = df[df.cls == c]
    normal = per_student_rate(g, g.window == "normal")
    for key, name in [("H4", "exam"), ("H5", "after")]:
        a = per_student_rate(g, g.window == name)
        o, lo, hi, p = paired_test(a, normal)
        R[key][c] = dict(diff=round(o, 2), lo=round(lo, 2), hi=round(hi, 2), p=p)
        print(f"{key} {c}: {name} days minus normal days = {o:.2f} points (95% range {lo:.2f} to {hi:.2f}, p = {p:.3f})")

# the 3 days after each exam, one exam at a time
by_exam = df[df.window == "after"].groupby(["exam_name", "cls"]).absent.apply(
    lambda s: round(100 * (1 - s.mean()), 2)).unstack(1)[CLASSES].loc[["Exam 1", "Exam 2", "1 Sem", "Final"]]
R["after_by_exam"] = {c: by_exam[c].to_dict() for c in CLASSES}
print("\nAttendance in the 3 days after each exam (%):\n", by_exam.to_string())

# ---- Rain (H3) ------------------------------------------------------------------
rain = df.groupby(["cls", "rainy"]).absent.apply(lambda s: round(100 * (1 - s.mean()), 2)).unstack(1)
R["rain"] = {c: {"dry": rain.loc[c, False], "rainy": rain.loc[c, True]} for c in CLASSES}
R["rainy_days"] = {c: int(df[(df.cls == c) & df.rainy].date.nunique()) for c in CLASSES}
print("\nAttendance on dry and rainy days (%):\n", rain.to_string())
R["H3"] = {}
for c in CLASSES:
    g = df[df.cls == c]
    o, lo, hi, p = paired_test(per_student_rate(g, g.rainy), per_student_rate(g, ~g.rainy))
    R["H3"][c] = dict(diff=round(o, 2), lo=round(lo, 2), hi=round(hi, 2), p=p)
    print(f"H3 {c}: rainy minus dry days = {o:.2f} points (95% range {lo:.2f} to {hi:.2f}, p = {p:.3f})")

# ---- Is a student's attendance steady through the year? ------------------------------
split = pd.Timestamp(SPLIT_DATE)
R["halves"] = {}
for c in CLASSES:
    g = df[df.cls == c]
    a, b = per_student_rate(g, g.date < split), per_student_rate(g, g.date >= split)
    R["halves"][c] = round(float(np.corrcoef(a, b)[0, 1]), 2)
print("\nCorrelation of attendance before and after", SPLIT_DATE, ":", R["halves"])

json.dump(R, open("results_step2.json", "w"), indent=1, default=str)
print("\nSaved fig1_months.png, fig2_exams.png and results_step2.json")
