import pandas as pd
import pytest
from analyze_fraud import score_transactions, summarize_results


# --- helpers ---

def make_scored(risk_labels, amounts, transaction_ids=None):
    n = len(risk_labels)
    return pd.DataFrame({
        "transaction_id": transaction_ids if transaction_ids else list(range(1, n + 1)),
        "account_id": list(range(1, n + 1)),
        "amount_usd": amounts,
        "risk_label": risk_labels,
    })


def make_chargebacks(*transaction_ids):
    return pd.DataFrame({"transaction_id": list(transaction_ids)})


# --- transaction counts ---

def test_transaction_counts_per_label():
    scored = make_scored(
        risk_labels=["low", "low", "high", "high", "high"],
        amounts=[10, 20, 100, 200, 300],
    )
    summary = summarize_results(scored, make_chargebacks())
    assert summary.loc[summary["risk_label"] == "low", "transactions"].iloc[0] == 2
    assert summary.loc[summary["risk_label"] == "high", "transactions"].iloc[0] == 3


# --- dollar metrics ---

def test_total_amount_usd_per_label():
    scored = make_scored(
        risk_labels=["low", "low", "high"],
        amounts=[100.0, 200.0, 500.0],
    )
    summary = summarize_results(scored, make_chargebacks())
    assert summary.loc[summary["risk_label"] == "low", "total_amount_usd"].iloc[0] == pytest.approx(300.0)
    assert summary.loc[summary["risk_label"] == "high", "total_amount_usd"].iloc[0] == pytest.approx(500.0)


def test_avg_amount_usd_per_label():
    scored = make_scored(
        risk_labels=["low", "low", "high", "high"],
        amounts=[100.0, 300.0, 400.0, 600.0],
    )
    summary = summarize_results(scored, make_chargebacks())
    assert summary.loc[summary["risk_label"] == "low", "avg_amount_usd"].iloc[0] == pytest.approx(200.0)
    assert summary.loc[summary["risk_label"] == "high", "avg_amount_usd"].iloc[0] == pytest.approx(500.0)


# --- chargeback rate ---

def test_chargeback_rate_when_all_confirmed_fraud():
    scored = make_scored(["high", "high"], [100.0, 200.0], transaction_ids=[10, 11])
    summary = summarize_results(scored, make_chargebacks(10, 11))
    assert summary.loc[summary["risk_label"] == "high", "chargeback_rate"].iloc[0] == pytest.approx(1.0)


def test_chargeback_rate_when_no_chargebacks():
    scored = make_scored(["low", "low"], [50.0, 75.0], transaction_ids=[1, 2])
    summary = summarize_results(scored, make_chargebacks())
    assert summary.loc[summary["risk_label"] == "low", "chargeback_rate"].iloc[0] == pytest.approx(0.0)


def test_chargeback_rate_partial():
    scored = make_scored(["high", "high", "high", "high"], [100.0] * 4, transaction_ids=[1, 2, 3, 4])
    summary = summarize_results(scored, make_chargebacks(1, 2))
    assert summary.loc[summary["risk_label"] == "high", "chargeback_rate"].iloc[0] == pytest.approx(0.5)


def test_non_chargeback_transaction_not_counted():
    scored = make_scored(["low", "high"], [100.0, 500.0], transaction_ids=[1, 2])
    # Only txn 2 (high) is a chargeback — txn 1 (low) should not count
    summary = summarize_results(scored, make_chargebacks(2))
    assert summary.loc[summary["risk_label"] == "low", "chargeback_rate"].iloc[0] == pytest.approx(0.0)
    assert summary.loc[summary["risk_label"] == "high", "chargeback_rate"].iloc[0] == pytest.approx(1.0)


def test_chargeback_rate_is_per_label_not_global():
    # 1 chargeback out of 2 high-risk, 0 out of 2 low-risk
    # Rate for high = 0.5, not 1/4 = 0.25
    scored = make_scored(["high", "high", "low", "low"], [100.0] * 4, transaction_ids=[1, 2, 3, 4])
    summary = summarize_results(scored, make_chargebacks(1))
    assert summary.loc[summary["risk_label"] == "high", "chargeback_rate"].iloc[0] == pytest.approx(0.5)


# --- score_transactions end-to-end ---

def test_score_transactions_produces_risk_columns():
    transactions = pd.DataFrame({
        "transaction_id": [1],
        "account_id": [10],
        "amount_usd": [1500.0],
        "device_risk_score": [80],
        "is_international": [1],
        "velocity_24h": [8],
        "failed_logins_24h": [6],
    })
    accounts = pd.DataFrame({
        "account_id": [10],
        "prior_chargebacks": [2],
    })
    result = score_transactions(transactions, accounts)
    assert "risk_score" in result.columns
    assert "risk_label" in result.columns
    assert result["risk_label"].iloc[0] == "high"


def test_score_transactions_low_risk_profile():
    transactions = pd.DataFrame({
        "transaction_id": [1],
        "account_id": [10],
        "amount_usd": [20.0],
        "device_risk_score": [5],
        "is_international": [0],
        "velocity_24h": [1],
        "failed_logins_24h": [0],
    })
    accounts = pd.DataFrame({
        "account_id": [10],
        "prior_chargebacks": [0],
    })
    result = score_transactions(transactions, accounts)
    assert result["risk_label"].iloc[0] == "low"
