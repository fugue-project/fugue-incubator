# flake8: noqa
import html
import json
from typing import Any, List, Dict

import fugue_sql
import pandas as pd
from fugue import (
    NativeExecutionEngine,
    make_execution_engine,
    register_execution_engine,
)
from fugue.extensions._builtins.outputters import Show
from IPython.core.magic import register_cell_magic
from IPython.display import HTML, Javascript, display
from triad import Schema, ParamDict
from triad.utils.convert import get_caller_global_local_vars


_HIGHLIGHT_JS = r"""
require(["codemirror/lib/codemirror"]);
function set(str) {
    var obj = {}, words = str.split(" ");
    for (var i = 0; i < words.length; ++i) obj[words[i]] = true;
    return obj;
  }
var fugue_keywords = "fill hash rand even presort persist broadcast params process output outtransform rowcount concurrency prepartition zip print title save append parquet csv json single checkpoint weak strong deterministic yield connect sample seed";
CodeMirror.defineMIME("text/x-mssql", {
    name: "sql",
    keywords: set(fugue_keywords + " add after all alter analyze and anti archive array as asc at between bucket buckets by cache cascade case cast change clear cluster clustered codegen collection column columns comment commit compact compactions compute concatenate cost create cross cube current current_date current_timestamp database databases datata dbproperties defined delete delimited deny desc describe dfs directories distinct distribute drop else end escaped except exchange exists explain export extended external false fields fileformat first following for format formatted from full function functions global grant group grouping having if ignore import in index indexes inner inpath inputformat insert intersect interval into is items join keys last lateral lazy left like limit lines list load local location lock locks logical macro map minus msck natural no not null nulls of on optimize option options or order out outer outputformat over overwrite partition partitioned partitions percent preceding principals purge range recordreader recordwriter recover reduce refresh regexp rename repair replace reset restrict revoke right rlike role roles rollback rollup row rows schema schemas select semi separated serde serdeproperties set sets show skewed sort sorted start statistics stored stratify struct table tables tablesample tblproperties temp temporary terminated then to touch transaction transactions transform true truncate unarchive unbounded uncache union unlock unset use using values view when where window with"),
    builtin: set("tinyint smallint int bigint boolean float double string binary timestamp decimal array map struct uniontype delimited serde sequencefile textfile rcfile inputformat outputformat"),
    atoms: set("false true null unknown"),
    operatorChars: /^[*\/+\-%<>!=&|^\/#@?~]/,
    dateSQL: set("datetime date time timestamp"),
    support: set("ODBCdotTable doubleQuote binaryNumber hexNumber commentSlashSlash commentHash")
  });
require(['notebook/js/codecell'], function(codecell) {
    codecell.CodeCell.options_default.highlight_modes['magic_text/x-mssql'] = {'reg':[/%%fsql/]} ;
    Jupyter.notebook.events.one('kernel_ready.Kernel', function(){
    Jupyter.notebook.get_cells().map(function(cell){
        if (cell.cell_type == 'code'){ cell.auto_highlight(); } }) ;
    });
  });
"""

_FUGUE_NOTEBOOK_PRE_CONF = ParamDict()
_FUGUE_NOTEBOOK_POST_CONF = ParamDict()


@register_cell_magic("fsql")
def fsql(line: str, cell: str) -> None:
    _, lc = get_caller_global_local_vars(start=-2, end=-2)
    line = line.strip()
    p = line.find("{")
    if p >= 0:
        engine = line[:p].strip()
        conf = json.loads(line[p:])
    else:
        parts = line.split(" ", 1)
        engine = parts[0]
        conf = ParamDict(None if len(parts) == 1 else lc[parts[1]])
    cf = dict(_FUGUE_NOTEBOOK_PRE_CONF)
    cf.update(conf)
    for k, v in _FUGUE_NOTEBOOK_POST_CONF.items():
        if k in cf and cf[k] != v:
            raise ValueError(
                f"{k} must be {v}, but you set to {cf[k]}, you may unset it"
            )
        cf[k] = v
    cf.update(_FUGUE_NOTEBOOK_POST_CONF)
    fugue_sql.fsql(cell).run(make_execution_engine(engine, cf))


class NotebookSetup(object):
    def get_pre_conf(self) -> Dict[str, Any]:
        return {"fugue.pre": 1}

    def get_post_conf(self) -> Dict[str, Any]:
        return {"fugue.post": 2}

    def pretty_print(
        self,
        schema: Schema,
        head_rows: List[List[Any]],
        title: Any,
        rows: int,
        count: int,
    ):
        components: List[Any] = []
        if title is not None:
            components.append(HTML(f"<h3>{html.escape(title)}</h3>"))
        pdf = pd.DataFrame(head_rows, columns=list(schema.names))
        components.append(pdf)
        if count >= 0:
            components.append(HTML(f"<strong>total count: {count}</strong>"))
        components.append(HTML(f"<small>schema: {schema}</small>"))
        display(*components)

    def register_execution_engines(self):
        register_execution_engine(
            "native", lambda conf, **kwargs: NativeExecutionEngine(conf=conf)
        )
        try:
            import pyspark  # noqa: F401
            from fugue_spark import SparkExecutionEngine

            register_execution_engine(
                "spark", lambda conf, **kwargs: SparkExecutionEngine(conf=conf)
            )
        except ImportError:
            pass
        try:
            import dask.dataframe  # noqa: F401
            from fugue_dask import DaskExecutionEngine

            register_execution_engine(
                "dask", lambda conf, **kwargs: DaskExecutionEngine(conf=conf)
            )
        except ImportError:
            pass

    def run(self) -> Any:
        _FUGUE_NOTEBOOK_PRE_CONF.clear()
        _FUGUE_NOTEBOOK_PRE_CONF.update(self.get_pre_conf())
        _FUGUE_NOTEBOOK_POST_CONF.clear()
        _FUGUE_NOTEBOOK_POST_CONF.update(self.get_post_conf())
        self.register_execution_engines()
        Show.set_hook(self.pretty_print)
        return Javascript(_HIGHLIGHT_JS)
