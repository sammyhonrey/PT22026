"""Presentation geometry must preserve zeroes and proportional evidence."""

from html.parser import HTMLParser
from unittest.mock import patch

import pandas as pd
import pytest

from src import ui
from src.viewport import layout_width


class MarkupReader(HTMLParser):
    def __init__(self, value):
        super().__init__()
        self.styles = []
        self.feed(value)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'i':
            self.styles.append(attrs.get('style'))


def test_recorded_bar_heights_preserve_zero_and_relative_values():
    frame = pd.DataFrame({'energy_kwh': [0.0, 0.5, 2.0] + [0.0] * 21},
                         index=pd.date_range('2026-01-01', periods=24, freq='h'))
    with patch('src.ui.html') as render:
        ui.recorded_bars(frame)
    styles = MarkupReader(render.call_args.args[0]).styles
    assert len(styles) == 24
    # The ribbon adds an hour index for the staggered reveal; geometry is unchanged.
    heights = [value.split(';')[0] for value in styles]
    assert heights[:3] == ['height:0.000000%', 'height:25.000000%', 'height:100.000000%']
    assert all(value == 'height:0.000000%' for value in heights[3:])


def test_original_metric_values_use_common_scale_without_mutating_input():
    metrics = pd.DataFrame({'model':['A','B','C'], 'mae_kwh':[.2,.1,.3], 'rmse_kwh':[.4,.2,.5]})
    original = metrics.copy(deep=True)
    with patch('src.ui.html') as render, patch('src.ui.st.caption'):
        ui.overview_evidence(metrics)
    markup = render.call_args.args[0]
    widths = [float(value.removeprefix('width:').removesuffix('%')) for value in MarkupReader(markup).styles]
    assert widths == pytest.approx([.1/.33*100, .2/.33*100, .3/.33*100])
    assert markup.index('<h3>B</h3>') < markup.index('<h3>A</h3>') < markup.index('<h3>C</h3>')
    pd.testing.assert_frame_equal(original, metrics)


@pytest.mark.parametrize('value,expected', [(1280,1280),(320,320),(None,0),(True,0),(-1,0),(16001,0),('320',0)])
def test_component_width_is_bounded_and_typed(value, expected):
    with patch('src.viewport._observe', return_value=value):
        assert layout_width() == expected


def test_diagnostic_component_is_not_an_invisible_keyboard_stop():
    """The width observer must not enter tab order or reserve a loading skeleton."""
    with patch('src.viewport._observe', return_value=320) as observe:
        assert layout_width() == 320
    observe.assert_called_once_with(key='ecowatt_layout_width', default=0, tab_index=-1, height=0)
