import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class RoutingBehaviorAnalyzer:
    def __init__(self):
        self.routing_data = {
            "Local": {
                "Classifier": 50,
                "Autoencoder": 50,
                "Description": "Balanced local routing"
            },
            "CICIDS2017": {
                "Classifier": 55,
                "Autoencoder": 45,
                "Description": "Classifier dominant"
            },
            "CIC-IDS2018": {
                "Classifier": 20,
                "Autoencoder": 80,
                "Description": "Heavy uncertainty routing"
            }
        }
        self.figures_dir = "results/figures"
        self.metrics_dir = "results/metrics"
        self.reports_dir = "results/reports"
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    def export_metrics(self):
        json_path = os.path.join(
            self.metrics_dir,
            "routing_behavior.json"
        )
        with open(json_path, "w") as f:
            json.dump(self.routing_data, f, indent=4)
        rows = []
        for dataset, values in self.routing_data.items():
            rows.append({
                "Dataset": dataset,
                "Classifier_Routing": values["Classifier"],
                "Autoencoder_Routing": values["Autoencoder"],
                "Dominant_Component":
                    "Classifier"
                    if values["Classifier"] >
                    values["Autoencoder"]
                    else "Autoencoder"
            })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(
            self.metrics_dir,
            "routing_behavior.csv"
        )
        df.to_csv(csv_path, index=False)
        logger.info(f"Metrics exported → {json_path}")
        logger.info(f"Metrics exported → {csv_path}")
        return df
    def compute_shift_score(self, classifier, autoencoder):
        return abs(classifier - autoencoder)
    def generate_interpretation(self, df):
        interpretation = []
        interpretation.append(
            "Routing behavior analysis reveals "
            "how prediction confidence changes "
            "across datasets."
        )
        for _, row in df.iterrows():
            dataset = row["Dataset"]
            clf = row["Classifier_Routing"]
            ae = row["Autoencoder_Routing"]
            if ae > clf:
                statement = (
                    f"{dataset} demonstrates "
                    f"strong uncertainty-aware routing "
                    f"({ae}% autoencoder vs "
                    f"{clf}% classifier), "
                    f"suggesting increased "
                    f"distributional complexity."
                )
            else:
                statement = (
                    f"{dataset} maintains relatively "
                    f"stable classifier confidence "
                    f"({clf}% classifier vs "
                    f"{ae}% autoencoder)."
                )
            interpretation.append(statement)
        interpretation.append(
            "Notably, CIC-IDS2018 exhibits "
            "substantially higher autoencoder routing, "
            "indicating that the hybrid system adapts "
            "under domain shift by delegating uncertain "
            "samples for reconstruction-based analysis."
        )
        report_path = os.path.join(
            self.reports_dir,
            "routing_interpretation.txt"
        )
        with open(report_path, "w",
                  encoding="utf-8") as f:
            f.write("\n\n".join(interpretation))
        logger.info(
            f"Interpretation saved → {report_path}"
        )
    def plot_routing_behavior(self):
        datasets = list(self.routing_data.keys())
        classifier_vals = [
            self.routing_data[d]["Classifier"]
            for d in datasets
        ]
        autoencoder_vals = [
            self.routing_data[d]["Autoencoder"]
            for d in datasets
        ]
        shift_scores = [
            self.compute_shift_score(c, a)
            for c, a in zip(
                classifier_vals,
                autoencoder_vals
            )
        ]
        x = np.arange(len(datasets))
        width = 0.35
        fig, ax = plt.subplots(
            figsize=(10, 7)
        )
        bars1 = ax.bar(
            x - width / 2,
            classifier_vals,
            width,
            label="Classifier Routing"
        )
        bars2 = ax.bar(
            x + width / 2,
            autoencoder_vals,
            width,
            label="Autoencoder Routing"
        )
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.0f}%",
                    xy=(
                        bar.get_x() +
                        bar.get_width() / 2,
                        height
                    ),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    fontweight="bold"
                )
        for i, score in enumerate(
                shift_scores):
            ax.text(
                i,
                102,
                f"Shift={score}",
                ha="center",
                fontsize=9,
                fontweight="bold"
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            datasets,
            fontweight="bold"
        )
        ax.set_ylabel(
            "Routing Percentage (%)",
            fontweight="bold"
        )
        ax.set_xlabel(
            "Dataset",
            fontweight="bold"
        )
        ax.set_title(
            "Routing Behavior Across Datasets",
            fontsize=14,
            fontweight="bold"
        )
        ax.set_ylim(0, 110)
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )
        ax.legend()
        figure_path = os.path.join(
            self.figures_dir,
            "routing_behavior.png"
        )
        plt.tight_layout()
        plt.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        logger.info(
            f"Figure saved → {figure_path}"
        )
    def run(self):
        logger.info(
            "Starting routing behavior analysis..."
        )
        df = self.export_metrics()
        self.plot_routing_behavior()
        self.generate_interpretation(df)
        logger.info(
            "Routing behavior analysis completed."
        )
if __name__ == "__main__":
    analyzer = RoutingBehaviorAnalyzer()
    analyzer.run()
