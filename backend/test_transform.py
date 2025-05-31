import pytest
from app import transform_payload_to_vector, feature_names

# ─────────────── Fixtures ──────────────────

@pytest.fixture
def sample_payload():
    return {
        "age": 35,
        "capital_gain": 1234,
        "capital_loss": 0,
        "hours_per_week": 40,
        "education_level": "education_prof_school",
        "marital_status": "marital_status_never_married",
        "occupation": "occupation_handlers_cleaners",
        "race": "race_asian_pac_islander",
        "relationship": "relationship_unmarried",
        "sex": "sex_female",
        "workclass": "workclass_govt_employees"
    }
    
@pytest.fixture
def numeric_only():
    return {
        "age": 50,
        "capital_gain": 200,
        "capital_loss": 10,
        "hours_per_week": 20
    }

# ─────────────── Tests ──────────────────

def test_vector_keys(sample_payload):
    vec = transform_payload_to_vector(sample_payload)
    assert set(vec.keys()) == set(feature_names)

def test_numeric_fields(sample_payload):
    vec = transform_payload_to_vector(sample_payload)
    assert vec["age"] == 35.0
    assert vec["capital.gain"] == 1234.0
    assert vec["capital.loss"] == 0.0
    assert vec["hours.per.week"] == 40.0

def test_one_hot_encoding(sample_payload):
    vec = transform_payload_to_vector(sample_payload)
    # Une seule éducation à 1
    edu_feats = [k for k in feature_names if k.startswith("education_")]
    assert sum(vec[k] for k in edu_feats) == 1
    assert vec["education_Prof-school"] == 1

    # Vérifie marital.status
    ms_feats = [k for k in feature_names if k.startswith("marital.status_")]
    assert sum(vec[k] for k in ms_feats) == 1
    assert vec["marital.status_Never-married"] == 1

def test_numeric_only(numeric_only):
    vec = transform_payload_to_vector(numeric_only)
    assert vec["age"] == 50.0
    assert vec["capital.gain"] == 200.0
    assert vec["capital.loss"] == 10.0
    assert vec["hours.per.week"] == 20.0
    binary_feats = [f for f in feature_names if "_" in f and f not in ["age","capital.gain","capital.loss","hours.per.week"]]
    assert all(vec[f] == 0 for f in binary_feats)

def test_none_category(sample_payload):
    p = sample_payload.copy()
    p["education_level"] = None
    vec = transform_payload_to_vector(p)
    edu_feats = [k for k in feature_names if k.startswith("education_")]
    assert sum(vec[k] for k in edu_feats) == 0

def test_unknown_value(sample_payload):
    p = sample_payload.copy()
    p["occupation"] = "occupation_unicorn_tamer"
    vec = transform_payload_to_vector(p)
    occ_feats = [k for k in feature_names if k.startswith("occupation_")]
    assert all(vec[k] == 0 for k in occ_feats)

def test_sex_male(sample_payload):
    p = sample_payload.copy()
    p["sex"] = "sex_male"
    vec = transform_payload_to_vector(p)
    assert vec["sex_Female"] == 0