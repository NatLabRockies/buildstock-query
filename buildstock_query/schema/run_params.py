from typing import Literal

from pydantic import BaseModel, ConfigDict


class RunParams(BaseModel):
    workgroup: str
    db_name: str
    table_name: str | tuple[str, str | None]
    buildstock_type: Literal["resstock", "comstock"] = 'resstock'
    db_schema: str | dict | None = None
    sample_weight_override: int | float | None = None
    region_name: str = "us-west-2"
    execution_history: str | None = None
    cache_folder: str = ".bsq_cache"
    athena_query_reuse: bool = True
    model_config = ConfigDict(arbitrary_types_allowed=True)
    keep_column_prefix: bool = False
    query_unload_s3_bucket: str = "resstock-core"


class BSQParams(RunParams):
    skip_reports: bool = False

    def get_run_params(self):
        return RunParams.model_validate(self.model_dump(include=set(RunParams.model_fields.keys())))
