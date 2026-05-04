"""
# BuildStockQuery
- - - - - - - - -
A library to run AWS Athena queries to get various data from a BuildStock run. The main class is called BuildStockQuery.
An object of BuildStockQuery needs to be created to perform various queries. In addition to supporting various
query member functions, the BuildStockQuery object contains 3 member objects that can be used to perform certain
class of queries and analysis. These member objects can be accessed as follows::

bsq = BuildStockQuery(...)  `BuildStockQuery` object
bsq.agg  `buildstock_query.aggregate_query.BuildStockAggregate`
bsq.report  `buildstock_query.report_query.BuildStockReport`
bsq.utility  `buildstock_query.utility_query.BuildStockUtility`

```
# The core query API lives on the BuildStockQuery object itself:
from buildstock_query import BuildStockQuery
bsq = BuildStockQuery(...)
bsq.get_results_csv()
bsq.get_upgrades_csv()
bsq.query(enduses=[...], annual_only=True, ...)                 # annual baseline / upgrade results
bsq.query(enduses=[...], annual_only=False, ...)                # timeseries aggregations
bsq.query(enduses=[...], upgrade_id="1", include_savings=True)  # savings shape

# Reports and utility-specific helpers:
bsq.report.get_success_report(...)
bsq.report.get_successful_simulation_count(...)
bsq.utility.aggregate_annual_by_eiaid(...)
```

In addition, the library also exposes `buildstock_query.tools.upgrades_analyzer.UpgradesAnalyzer`. It can be used to
perform quality check for the apply logic in buildstock configuration file.
```
from buildstock_query import UpgradesAnalyzer
ua = UpgradesAnalyzer(yaml_file='my_buildstock_configuration.yml', 'my_buildstock.csv')
options_report = ua.get_report()
options_report.drop(columns=['applicable_buildings']).to_csv('options_report.csv')
ua.save_detailed_report('detailed_report.csv')
```

`buildstock_query.tools.upgrades_analyzer.UpgradesAnalyzer` is also exposed as an script and can be directly used
from the command line by simply calling it (from the env buildstock_query is installed in):
```
>>>upgrades_analyzer
Welcome to upgrades analyzer
...
```

There is also another experimental tool called `buildstock_query.tools.upgrades_visualizer` available from command line.
The tool starts a localhost poltly dash dashboard that can be used for analytic visualization of annual results for
different upgrades.
```
>>>upgrades_visualizer
Welcome to upgrades visualizer
...
```
"""  # noqa: W291
from buildstock_query.schema.utilities import MappedColumn
from buildstock_query.query_core import ExeId
from buildstock_query.main import BuildStockQuery
from buildstock_query.tools import UpgradesAnalyzer
from buildstock_query.helpers import KWH2MBTU
from buildstock_query.helpers import MBTU2KWH
__all__ = ['BuildStockQuery', 'UpgradesAnalyzer', 'KWH2MBTU', 'MBTU2KWH', 'ExeId', 'MappedColumn']
