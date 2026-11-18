import datetime
from typing import Any, cast

from langfuse.api import (
    Dataset,
    DatasetItem,
    LangfuseAPI,
    NotFoundError,
    PaginatedDatasetItems,
    Projects,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from spendguard.observability.tracing import Tracing, create_tracing
from spendguard.settings import Settings


class FakeLangfuseProjects:
    def get(self, **keyword_arguments: Any) -> Projects:
        return Projects.model_validate(
            {
                'data': [
                    {
                        'id': 'project-1',
                        'name': 'SpendGuard',
                        'organization': {'id': 'organisation-1', 'name': 'Halden'},
                        'metadata': {},
                    }
                ]
            }
        )


class FakeLangfuseDatasets:
    def __init__(self) -> None:
        self.created: dict[str, Dataset] = {}

    def get(self, dataset_name: str, **keyword_arguments: Any) -> Dataset:
        if dataset_name not in self.created:
            raise NotFoundError(body={'message': 'Dataset not found'})
        return self.created[dataset_name]

    def create(self, *, name: str, **keyword_arguments: Any) -> Dataset:
        moment = datetime.datetime(2026, 11, 20, tzinfo=datetime.UTC).isoformat()
        self.created[name] = Dataset.model_validate(
            {
                'id': f'dataset-{len(self.created) + 1}',
                'name': name,
                'metadata': None,
                'project_id': 'project-1',
                'created_at': moment,
                'updated_at': moment,
            }
        )
        return self.created[name]


class FakeLangfuseDatasetItems:
    def __init__(self, datasets: FakeLangfuseDatasets) -> None:
        self.datasets = datasets
        self.items: dict[str, DatasetItem] = {}

    def create(
        self, *, dataset_name: str, id: str, input: Any, metadata: Any, **keyword_arguments: Any
    ) -> DatasetItem:
        moment = datetime.datetime(2026, 11, 20, tzinfo=datetime.UTC).isoformat()
        self.items[id] = DatasetItem.model_validate(
            {
                'id': id,
                'status': 'ACTIVE',
                'input': input,
                'expected_output': keyword_arguments.get('expected_output'),
                'metadata': metadata,
                'dataset_id': self.datasets.get(dataset_name).id,
                'dataset_name': dataset_name,
                'media_references': [],
                'created_at': moment,
                'updated_at': moment,
            }
        )
        return self.items[id]

    def list(self, *, dataset_name: str, **keyword_arguments: Any) -> PaginatedDatasetItems:
        items = [item for item in self.items.values() if item.dataset_name == dataset_name]
        return PaginatedDatasetItems.model_validate(
            {
                'data': [item.model_dump() for item in items],
                'meta': {'page': 1, 'limit': 50, 'total_items': len(items), 'total_pages': 1},
            }
        )


class FakeLangfuseApi:
    def __init__(self) -> None:
        self.projects = FakeLangfuseProjects()
        self.datasets = FakeLangfuseDatasets()
        self.dataset_items = FakeLangfuseDatasetItems(self.datasets)


def attribute(span: ReadableSpan, name: str) -> str:
    return str((span.attributes or {})[name])


def trace_id_of(span: ReadableSpan) -> int:
    assert span.context is not None
    return int(span.context.trace_id)


def tracing_into(
    exporter: InMemorySpanExporter, langfuse_api: FakeLangfuseApi, settings: Settings
) -> Tracing:
    tracing = create_tracing(settings, span_exporter=exporter, tracer_provider=TracerProvider())
    assert tracing is not None
    tracing.client.api = cast(LangfuseAPI, langfuse_api)
    return tracing
