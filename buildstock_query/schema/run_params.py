from typing import Optional, Union
from typing import Literal
from pydantic import ConfigDict, BaseModel


class RunParams(BaseModel):
    workgroup: str
    db_name: str
    table_name: Union[str, tuple[str, Optional[str], Optional[str]]]
    buildstock_type: Literal["resstock", "comstock"] = 'resstock'
    db_schema: Optional[str | dict] = None
    sample_weight_override: Optional[Union[int, float]] = None
    region_name: str = "us-west-2"
    execution_history: Optional[str] = None
    cache_folder: str = ".bsq_cache"
    athena_query_reuse: bool = True
    model_config = ConfigDict(arbitrary_types_allowed=True)
    keep_column_prefix: bool = False
    query_unload_s3_bucket: str = "resstock-core"


class BSQParams(RunParams):
    skip_reports: bool = False

    def get_run_params(self):
        return RunParams.model_validate(self.model_dump(include=set(RunParams.model_fields.keys())))
