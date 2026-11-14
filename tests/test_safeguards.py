import pytest


@pytest.fixture(autouse=True)
def _inner_run_configuration(pytester: pytest.Pytester) -> None:
    pytester.makeini('[pytest]\nasyncio_default_fixture_loop_scope = function\n')


def test_a_run_with_a_skipped_test_fails_even_when_everything_else_passes(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        """
        import pytest

        def test_passes():
            pass

        def test_is_skipped():
            pytest.skip('service missing')
        """
    )

    result = pytester.runpytest('-p', 'safeguards')

    result.assert_outcomes(passed=1, skipped=1)
    assert result.ret != 0


def test_a_run_where_every_test_passes_succeeds(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_passes():
            pass
        """
    )

    result = pytester.runpytest('-p', 'safeguards')

    result.assert_outcomes(passed=1)
    assert result.ret == 0
