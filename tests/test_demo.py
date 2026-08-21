"""The CLI entrypoint runs end-to-end and prints the expected report."""
from raters import demo


def test_demo_main_runs(capsys):
    rc = demo.main(["--epochs", "80", "--raters", "3", "--seed", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Epoch staging agreement" in out
    assert "Fleiss' kappa" in out
    assert "ICC(2,1) absolute agreement" in out
    assert "ICC(3,1) consistency" in out


def test_demo_two_raters(capsys):
    rc = demo.main(["--epochs", "60", "--raters", "2", "--seed", "5"])
    assert rc == 0
    assert "2 raters" in capsys.readouterr().out
