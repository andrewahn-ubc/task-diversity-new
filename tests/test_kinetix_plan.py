import pytest

from td2_kinetix.plan import (
    DATA_TASKS,
    HELDOUT_TASKS,
    MAIN_TASKS,
    STREAMS,
    data_task,
    experiment_task,
    main,
    stream_task,
)


def test_plan_counts_and_boundaries():
    assert (DATA_TASKS, STREAMS, MAIN_TASKS, HELDOUT_TASKS) == (22, 15, 105, 105)
    assert data_task(0) == {"kind": "train", "seed": 0, "distribution": 0}
    assert data_task(20) == {"kind": "train", "seed": 2, "distribution": 6}
    assert data_task(21) == {"kind": "heldout", "seed": None, "distribution": None}
    assert stream_task(0) == {"diversity": 1, "seed": 0}
    assert stream_task(14) == {"diversity": 65_536, "seed": 2}
    assert experiment_task(104) == {
        "distribution": 6,
        "diversity": 65_536,
        "seed": 2,
        "stream_index": 14,
    }


@pytest.mark.parametrize("fn,index", [(data_task, -1), (data_task, 22), (stream_task, 15), (experiment_task, 105)])
def test_plan_rejects_out_of_range(fn, index):
    with pytest.raises(IndexError):
        fn(index)


def test_shell_friendly_plan_aliases(capsys):
    assert main(["stream", "--index", "14"]) == 0
    assert capsys.readouterr().out == "65536\t2\n"
    assert main(["data", "--index", "21"]) == 0
    assert capsys.readouterr().out == "heldout\t\t\n"
    assert main(["identity"]) == 0
    assert len(capsys.readouterr().out.strip()) == 64
