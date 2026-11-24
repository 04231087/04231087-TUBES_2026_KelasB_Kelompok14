import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GA Optimization", layout="wide")
st.title(" Minggu 13 — Genetic Algorithm: Seleksi Channel EEG")

# ---------- lokasi data: folder `data/` di root repo (atau root repo langsung) ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" if (ROOT / "data").exists() else ROOT


def read_csv(name):
    p = DATA / name
    return pd.read_csv(p) if p.exists() else None


runs = []
for p in sorted(DATA.glob("ga_run_*.json")):
    with open(p) as f:
        runs.append(json.load(f))

if not runs:
    st.error(
        "File hasil GA belum ditemukan. Upload `ga_run_*.json`, `ga_per_seed.csv`, "
        "`ga_features_before_after.csv`, dan `w13_baseline_vs_ga.csv` dari "
        "`w13_hasil_ga.zip` ke folder `data/` di repo."
    )
    st.stop()

params = runs[0]["params"]

# ---------- 1. Desain GA ----------
st.header("1. Desain GA")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Encoding & fitness**")
    st.markdown(
        "- Kromosom: 14 bit biner (1 bit per channel EEG; 1 = channel dipakai)\n"
        "- Individu 0 pada populasi awal = semua channel (jaminan baseline ada di populasi)\n"
        "- Kromosom kosong diperbaiki dengan mengaktifkan 1 bit acak"
    )
    st.latex(r"\text{Fitness} = F1_{val} - \alpha \cdot \frac{n_{fitur}}{14}")
    st.caption(
        "F1 dihitung pada **validation set** (split kronologis); test set tidak dipakai "
        "dalam pencarian. Suku α memberi penalti pada jumlah fitur."
    )
with c2:
    st.markdown("**Hyperparameter GA**")
    hp = pd.DataFrame(
        {
            "Parameter": [
                "Ukuran populasi", "Generasi (stopping criterion)", "Seleksi",
                "Crossover (uniform)", "Mutasi (bit-flip)", "Elitism", "α (penalti fitur)",
                "Epoch LSTM per evaluasi", "Seq len",
            ],
            "Nilai": [
                params["pop_size"], params["generations"], f"Tournament k={params['tournament_k']}",
                f"rate {params['crossover_rate']}", f"rate {params['mutation_rate']}/bit",
                params["elitism"], params["alpha"], params["epochs"], params["seq_len"],
            ],
        }
    )
    st.dataframe(hp, hide_index=True, use_container_width=True)
st.info(f"Jumlah independent run (seed): **{len(runs)}** — seed {[r['seed'] for r in runs]}")

# ---------- 2. Konvergensi ----------
st.header("2. Convergence curve")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for r in runs:
    gen = [h["gen"] for h in r["history"]]
    ax[0].plot(gen, [h["best"] for h in r["history"]], marker="o", label=f"seed {r['seed']}")
    ax[1].plot(gen, [h["mean"] for h in r["history"]], marker="o", label=f"seed {r['seed']}")
ax[0].set_title("Best fitness per generasi")
ax[1].set_title("Mean fitness populasi")
for a in ax:
    a.set_xlabel("Generasi")
    a.set_ylabel("Fitness")
    a.grid(alpha=0.3)
    a.legend()
plt.tight_layout()
st.pyplot(fig)

# ---------- 3. Fitur sebelum / sesudah ----------
st.header("3. Fitur sebelum & sesudah GA")
feat = read_csv("ga_features_before_after.csv")
if feat is not None:
    st.dataframe(feat, hide_index=True, use_container_width=True)
    sel = (feat["Sesudah GA (run terbaik)"] == "✓").sum()
    st.success(f"Channel dipertahankan (run terbaik): **{sel} dari {len(feat)}**")
else:
    st.warning("`ga_features_before_after.csv` belum ada di folder data.")

# ---------- 4. Ringkasan per seed (reproducibility) ----------
st.header("4. Hasil per seed (keterulangan)")
per_seed = read_csv("ga_per_seed.csv")
if per_seed is not None:
    st.dataframe(per_seed, hide_index=True, use_container_width=True)
    fits = [r["best_fitness"] for r in runs]
    s = pd.Series(fits)
    st.write(
        f"Best fitness antar seed: mean **{s.mean():.4f}**, std **{s.std(ddof=0):.4f}**, "
        f"min {s.min():.4f}, max {s.max():.4f}"
    )
else:
    st.warning("`ga_per_seed.csv` belum ada di folder data.")

# ---------- 5. Baseline vs GA+NN ----------
st.header("5. Baseline NN vs GA+NN")
cmp_ = read_csv("w13_baseline_vs_ga.csv")
if cmp_ is not None:
    st.dataframe(cmp_, hide_index=True, use_container_width=True)
    st.caption("Metrik pada validation set; waktu = waktu training per seed dan waktu pencarian GA.")
else:
    st.warning("`w13_baseline_vs_ga.csv` belum ada di folder data.")

st.caption("Protokol: split kronologis tetap, scaler dari data latih, seleksi fitur hanya berdasarkan validation set.")
