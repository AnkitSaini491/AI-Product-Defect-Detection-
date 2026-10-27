

import pandas as pd
import plotly.express as px
from pathlib import Path


def analyze_defects(defects):
    """Analyze detected defects and generate quality statistics."""

    df = pd.DataFrame(defects)

    if df.empty:
        return {
            "total_defects": 0,
            "defect_types": 0,
            "defect_counts": pd.DataFrame(
                columns=["Defect Type", "Count"]
            ),
            "average_confidence": 0
        }

    # Count defects by category
    defect_counts = (
        df["defect_type"]
        .value_counts()
        .reset_index()
    )

    defect_counts.columns = ["Defect Type", "Count"]

    return {
        "total_defects": len(df),
        "defect_types": df["defect_type"].nunique(),
        "defect_counts": defect_counts,
        "average_confidence": round(
            df["confidence"].mean(), 2
        ) if "confidence" in df.columns else 0
    }


def calculate_quality(total_products, defective_products):
    """Calculate product quality metrics."""

    if total_products <= 0:
        raise ValueError("Total products must be greater than zero.")

    if not 0 <= defective_products <= total_products:
        raise ValueError(
            "Defective products must be between zero "
            "and total products."
        )

    defect_rate = (
        defective_products / total_products
    ) * 100

    good_products = total_products - defective_products

    return {
        "total_products": total_products,
        "defective_products": defective_products,
        "good_products": good_products,
        "defect_rate": round(defect_rate, 2),
        "quality_rate": round(100 - defect_rate, 2)
    }


def create_defect_chart(defect_counts):
    """Create a bar chart of defect categories."""

    if defect_counts.empty:
        return None

    fig = px.bar(
        defect_counts,
        x="Defect Type",
        y="Count",
        title="Product Defects by Category",
        text="Count"
    )

    fig.update_layout(
        xaxis_title="Defect Type",
        yaxis_title="Number of Defects"
    )

    return fig


def save_analysis(defects, output_file="output/defect_analysis.csv"):
    """Save defect records to a CSV file."""

    df = pd.DataFrame(defects)

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(output_path, index=False)

    return str(output_path)


if __name__ == "__main__":

    # Sample inspection records
    sample_defects = [
        {"defect_type": "Scratch", "confidence": 92.5},
        {"defect_type": "Crack", "confidence": 88.2},
        {"defect_type": "Scratch", "confidence": 95.1},
        {"defect_type": "Dent", "confidence": 84.7}
    ]

    analysis = analyze_defects(sample_defects)

    print("Total Defects:", analysis["total_defects"])
    print("Defect Categories:", analysis["defect_types"])
    print(
        "Average Confidence:",
        analysis["average_confidence"], "%"
    )

    print("\nDefect Counts:")
    print(analysis["defect_counts"])

    quality = calculate_quality(
        total_products=100,
        defective_products=12
    )

    print("\nQuality Report:")
    print("Total Products:", quality["total_products"])
    print("Good Products:", quality["good_products"])
    print("Defective Products:", quality["defective_products"])
    print("Defect Rate:", quality["defect_rate"], "%")
    print("Quality Rate:", quality["quality_rate"], "%")

    chart = create_defect_chart(
        analysis["defect_counts"]
    )

    if chart is not None:
        chart.show()

    save_analysis(sample_defects)
    print("\nAnalysis saved successfully!")
