from app.data.generator import generate_synthetic_data

def test_synthetic_dataset():
    df = generate_synthetic_data("/tmp/econosim_test.csv", n_rows=100)
    assert len(df) == 100
    assert "gdp" in df.columns
    assert "inflation" in df.columns
