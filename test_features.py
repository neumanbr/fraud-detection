import pandas as pd
import pytest
from features import build_model_frame


def make_inputs(amount_usd=500, failed_logins_24h=0, prior_chargebacks=0):
    transactions = pd.DataFrame({
        "transaction_id": [1],
        "account_id": [100],
        "amount_usd": [amount_usd],
        "failed_logins_24h": [failed_logins_24h],
    })
    accounts = pd.DataFrame({
        "account_id": [100],
        "prior_chargebacks": [prior_chargebacks],
    })
    return transactions, accounts


# --- is_large_amount ---

def test_is_large_amount_below_threshold():
    tx, acc = make_inputs(amount_usd=999)
    result = build_model_frame(tx, acc)
    assert result["is_large_amount"].iloc[0] == 0


def test_is_large_amount_at_threshold():
    tx, acc = make_inputs(amount_usd=1000)
    result = build_model_frame(tx, acc)
    assert result["is_large_amount"].iloc[0] == 1


def test_is_large_amount_above_threshold():
    tx, acc = make_inputs(amount_usd=2500)
    result = build_model_frame(tx, acc)
    assert result["is_large_amount"].iloc[0] == 1


# --- login_pressure ---

def test_login_pressure_none():
    tx, acc = make_inputs(failed_logins_24h=0)
    result = build_model_frame(tx, acc)
    assert result["login_pressure"].iloc[0] == "none"


@pytest.mark.parametrize("n", [1, 2])
def test_login_pressure_low(n):
    tx, acc = make_inputs(failed_logins_24h=n)
    result = build_model_frame(tx, acc)
    assert result["login_pressure"].iloc[0] == "low"


def test_login_pressure_high():
    tx, acc = make_inputs(failed_logins_24h=3)
    result = build_model_frame(tx, acc)
    assert result["login_pressure"].iloc[0] == "high"


# --- account join ---

def test_prior_chargebacks_merged_from_accounts():
    tx, acc = make_inputs(prior_chargebacks=3)
    result = build_model_frame(tx, acc)
    assert result["prior_chargebacks"].iloc[0] == 3


def test_all_transactions_retained_after_join():
    transactions = pd.DataFrame({
        "transaction_id": [1, 2, 3],
        "account_id": [100, 100, 100],
        "amount_usd": [100, 200, 300],
        "failed_logins_24h": [0, 1, 2],
    })
    accounts = pd.DataFrame({
        "account_id": [100],
        "prior_chargebacks": [0],
    })
    result = build_model_frame(transactions, accounts)
    assert len(result) == 3
