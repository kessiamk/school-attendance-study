# step3_ml.py
# Run this AFTER step2_analysis.py, in the same Colab notebook.
# Two machine learning tasks: (1) group students with k-means, (2) early warning of chronic absence.

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

R3 = {}
split = pd.Timestamp(SPLIT_DATE)

# ---- 1. k-means: group students by how they are absent -----------------------------
# Four measures for every student (all from the register, nothing else):
#   year absence %, change in absence % (second half minus first half of the year),
#   average length of an absence spell (days), absence % on the first day after a holiday.
rows = []
for (c, s), g in df.groupby(["cls", "student"]):
    g = g.sort_values("date")
    sp = spell_lengths(g.absent.tolist())
    rows.append(dict(cls=c, student=s,
                     year_absence=100 * g.absent.mean(),
                     change=100 * (g[g.date >= split].absent.mean() - g[g.date < split].absent.mean()),
                     spell=float(np.mean(sp)) if sp else 0.0,
                     after_holiday=100 * g[g.after_holiday].absent.mean()))
F = pd.DataFrame(rows)
FEATS = ["year_absence", "change", "spell", "after_holiday"]
X = StandardScaler().fit_transform(F[FEATS])

sil = {}
for k in range(2, 7):
    sil[k] = silhouette_score(X, KMeans(k, n_init=10, random_state=42).fit(X).labels_)
best_k = max(sil, key=sil.get)
print("Silhouette score for 2 to 6 groups (higher = cleaner groups):",
      {k: round(v, 3) for k, v in sil.items()}, "-> best:", best_k)
R3["silhouette"] = {str(k): round(v, 3) for k, v in sil.items()}

F["group"] = KMeans(best_k, n_init=10, random_state=42).fit(X).labels_
# name the groups by their average attendance: Group A = highest attendance
order = F.groupby("group").year_absence.mean().sort_values().index.tolist()
F["group"] = F.group.map({g: "ABCDEF"[i] for i, g in enumerate(order)})

summary = F.groupby("group").agg(
    students=("student", "size"),
    class_VIII=("cls", lambda s: int((s == "Class VIII").sum())),
    class_IX=("cls", lambda s: int((s == "Class IX").sum())),
    attendance=("year_absence", lambda s: round(100 - s.mean(), 1)),
    absent_after_holiday=("after_holiday", lambda s: round(s.mean(), 1)),
    spell_days=("spell", lambda s: round(s.mean(), 1)),
).reset_index()
R3["groups"] = summary.to_dict("records")
print("\nGroups found by k-means:\n", summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6.75))
gcol = dict(zip("ABCDEF", ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]))
for g in sorted(F.group.unique()):
    for c, mk in [("Class VIII", "o"), ("Class IX", "^")]:
        sub = F[(F.group == g) & (F.cls == c)]
        if len(sub):
            ax.scatter(100 - sub.year_absence, sub.after_holiday, s=90, marker=mk, color=gcol[g],
                       edgecolor="black", alpha=0.85, label=f"Group {g}, {c}")
ax.axvline(90, color="grey", ls=":", lw=1.5)
ax.axvline(75, color="red", ls="--", lw=1.2)
ax.text(90.2, ax.get_ylim()[1] * 0.93, "90% line", color="grey")
ax.text(75.2, ax.get_ylim()[1] * 0.93, "75% line", color="red")
ax.set_xlabel("Attendance for the year (%)")
ax.set_ylabel("Absent on the first day after a holiday (% of those days)")
ax.set_title("Students grouped by k-means (each dot = one student, numbered only)")
ax.legend(loc="upper right", title="Colour = group, shape = class")
fig.tight_layout()
fig.savefig("fig3_groups.png", dpi=100)
plt.close(fig)

# ---- 2. Early warning: can the first four weeks show who will be chronically absent? ----
# Class VIII only (no Class IX student fell below 90%).
if "Class VIII" in df.cls.values:
    rows = []
    for s, g in df[df.cls == "Class VIII"].groupby("student"):
        g = g.sort_values("date")
        early, rest = g.iloc[:EARLY_DAYS], g.iloc[EARLY_DAYS:]
        rows.append((s, 100 * early.absent.mean(), 100 * rest.absent.mean()))
    E = pd.DataFrame(rows, columns=["student", "early_absence", "later_absence"])
    E["chronic_later"] = (100 - E.later_absence < CHRONIC_BELOW).astype(int)
    n, k = len(E), int(E.chronic_later.sum())

    flag = (E.early_absence >= 10).astype(int)            # the simple rule: 10% or more absent in weeks 1-4
    tp = int(((flag == 1) & (E.chronic_later == 1)).sum())
    fp = int(((flag == 1) & (E.chronic_later == 0)).sum())
    fn = int(((flag == 0) & (E.chronic_later == 1)).sum())
    tn = int(((flag == 0) & (E.chronic_later == 0)).sum())
    acc = (tp + tn) / n
    base = max(k, n - k) / n                              # baseline: give every student the commonest answer
    lo, hi = wilson_interval(tp + tn, n)
    plo, phi = wilson_interval(tp, tp + fp)

    # logistic regression, leaving each student out in turn
    pred = []
    for i in range(n):
        tr = E.drop(index=i)
        m = LogisticRegression().fit(tr[["early_absence"]], tr.chronic_later)
        pred.append(int(m.predict(E.loc[[i], ["early_absence"]])[0]))
    lr_acc = float(np.mean(np.array(pred) == E.chronic_later.values))

    R3["early"] = dict(n=n, chronic_later=k, tp=tp, fp=fp, fn=fn, tn=tn, accuracy=round(acc, 3),
                       acc_lo=round(lo, 3), acc_hi=round(hi, 3), baseline=round(base, 3),
                       recall=round(tp / k, 3), precision=round(tp / (tp + fp), 3),
                       prec_lo=round(plo, 3), prec_hi=round(phi, 3),
                       unflagged_chronic=round(fn / (fn + tn), 3),
                       logistic_accuracy=round(lr_acc, 3), logistic_flagged=int(sum(pred)),
                       r=round(float(np.corrcoef(E.early_absence, E.later_absence)[0, 1]), 2))
    print("\nEarly warning, Class VIII:")
    print(f"  Rule 'absent 10% or more in weeks 1-4': flagged {tp + fp} students; {tp} became chronically absent, "
          f"{fp} did not; {fn} were missed; {tn} correctly not flagged")
    print(f"  Accuracy {acc:.1%} (95% range {lo:.1%} to {hi:.1%}); baseline of giving everyone the commonest answer {base:.1%}")
    print(f"  Of students flagged, {tp / (tp + fp):.0%} became chronic; of students not flagged, {fn / (fn + tn):.0%} did")
    print(f"  Caught {tp} of {k} later-chronic students ({tp / k:.0%})")
    print(f"  Logistic regression (leave one out): accuracy {lr_acc:.1%}, flagged {sum(pred)} of {n} students")
    print(f"  Correlation of early and later absence: r = {R3['early']['r']}")

    fig, ax = plt.subplots(figsize=(12, 6.75))
    for v, colr, lab in [(1, "#d62728", "Became chronically absent later (attendance below 90%)"),
                         (0, "#1f77b4", "Did not become chronically absent")]:
        sub = E[E.chronic_later == v]
        ax.scatter(sub.early_absence, sub.later_absence, s=90, color=colr, edgecolor="black", alpha=0.85, label=lab)
    ax.axvline(10, color="black", ls="--", lw=1.2, label="Rule: 10% absent in the first four weeks")
    ax.axhline(10, color="grey", ls=":", lw=1.5, label="10% absent in the rest of the year")
    ax.set_xlabel("Absent in the first four weeks (% of days)")
    ax.set_ylabel("Absent in the rest of the year (% of days)")
    ax.set_title("Class VIII: does the first month show who will be absent later? (each dot = one student)")
    ax.legend(loc="lower right", title="Colour = what happened later")
    fig.tight_layout()
    fig.savefig("fig4_early_warning.png", dpi=100)
    plt.close(fig)

json.dump(R3, open("results_step3.json", "w"), indent=1, default=str)
print("\nSaved fig3_groups.png, fig4_early_warning.png and results_step3.json")
