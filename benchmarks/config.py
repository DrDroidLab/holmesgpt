#!/usr/bin/env python3
"""
Configuration and Credentials Management

Loads credentials from a unified YAML file for all infrastructure and monitoring tools.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

BENCHMARKS_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
CONFIG_DIR = BENCHMARKS_DIR / "config"
RESULTS_DIR = BENCHMARKS_DIR / "results"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "llm" / "fixtures" / "test_ask_holmes"

# Default credentials file location
DEFAULT_CREDENTIALS_FILE = CONFIG_DIR / "credentials.yaml"


@dataclass
class KubernetesCredentials:
    """Kubernetes cluster credentials."""

    kubeconfig: Optional[str] = None
    context: Optional[str] = None
    namespace: Optional[str] = None


@dataclass
class DatadogCredentials:
    """Datadog credentials."""

    api_key: Optional[str] = None
    app_key: Optional[str] = None
    site: str = "datadoghq.com"


@dataclass
class NewRelicCredentials:
    """New Relic credentials."""

    api_key: Optional[str] = None
    account_id: Optional[str] = None
    region: str = "US"


@dataclass
class PrometheusCredentials:
    """Prometheus credentials."""

    url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class GrafanaCredentials:
    """Grafana credentials."""

    url: Optional[str] = None
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class LokiCredentials:
    """Loki credentials."""

    url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class ElasticsearchCredentials:
    """Elasticsearch/OpenSearch credentials."""

    url: Optional[str] = None
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class OpenAICredentials:
    """OpenAI/LLM credentials."""

    api_key: Optional[str] = None
    org_id: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class JudgeConfig:
    """LLM Judge configuration."""

    model: str = "gpt-4.1"
    api_key: Optional[str] = None


@dataclass
class Credentials:
    """All credentials for benchmark testing."""

    kubernetes: KubernetesCredentials = field(default_factory=KubernetesCredentials)
    datadog: DatadogCredentials = field(default_factory=DatadogCredentials)
    newrelic: NewRelicCredentials = field(default_factory=NewRelicCredentials)
    prometheus: PrometheusCredentials = field(default_factory=PrometheusCredentials)
    grafana: GrafanaCredentials = field(default_factory=GrafanaCredentials)
    loki: LokiCredentials = field(default_factory=LokiCredentials)
    elasticsearch: ElasticsearchCredentials = field(default_factory=ElasticsearchCredentials)
    openai: OpenAICredentials = field(default_factory=OpenAICredentials)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    custom: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "Credentials":
        """Load credentials from YAML file."""
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            kubernetes=KubernetesCredentials(**data.get("kubernetes", {})),
            datadog=DatadogCredentials(**data.get("datadog", {})),
            newrelic=NewRelicCredentials(**data.get("newrelic", {})),
            prometheus=PrometheusCredentials(**data.get("prometheus", {})),
            grafana=GrafanaCredentials(**data.get("grafana", {})),
            loki=LokiCredentials(**data.get("loki", {})),
            elasticsearch=ElasticsearchCredentials(**data.get("elasticsearch", {})),
            openai=OpenAICredentials(**data.get("openai", {})),
            judge=JudgeConfig(**data.get("judge", {})),
            custom=data.get("custom", {}),
        )

    def to_env_vars(self) -> Dict[str, str]:
        """Convert credentials to environment variables."""
        env_vars = {}

        # Kubernetes
        if self.kubernetes.kubeconfig:
            env_vars["KUBECONFIG"] = self.kubernetes.kubeconfig
        if self.kubernetes.context:
            env_vars["KUBE_CONTEXT"] = self.kubernetes.context

        # Datadog
        if self.datadog.api_key:
            env_vars["DD_API_KEY"] = self.datadog.api_key
        if self.datadog.app_key:
            env_vars["DD_APP_KEY"] = self.datadog.app_key
        if self.datadog.site:
            env_vars["DD_SITE"] = self.datadog.site

        # New Relic
        if self.newrelic.api_key:
            env_vars["NEW_RELIC_API_KEY"] = self.newrelic.api_key
        if self.newrelic.account_id:
            env_vars["NEW_RELIC_ACCOUNT_ID"] = self.newrelic.account_id

        # Prometheus
        if self.prometheus.url:
            env_vars["PROMETHEUS_URL"] = self.prometheus.url
        if self.prometheus.username:
            env_vars["PROMETHEUS_USERNAME"] = self.prometheus.username
        if self.prometheus.password:
            env_vars["PROMETHEUS_PASSWORD"] = self.prometheus.password

        # Grafana
        if self.grafana.url:
            env_vars["GRAFANA_URL"] = self.grafana.url
        if self.grafana.api_key:
            env_vars["GRAFANA_API_KEY"] = self.grafana.api_key

        # Loki
        if self.loki.url:
            env_vars["LOKI_URL"] = self.loki.url

        # Elasticsearch
        if self.elasticsearch.url:
            env_vars["ELASTICSEARCH_URL"] = self.elasticsearch.url
        if self.elasticsearch.api_key:
            env_vars["ELASTICSEARCH_API_KEY"] = self.elasticsearch.api_key

        # OpenAI
        if self.openai.api_key:
            env_vars["OPENAI_API_KEY"] = self.openai.api_key
        if self.openai.org_id:
            env_vars["OPENAI_ORG_ID"] = self.openai.org_id
        if self.openai.base_url:
            env_vars["OPENAI_BASE_URL"] = self.openai.base_url

        return env_vars

    def apply_to_env(self) -> None:
        """Apply credentials to current environment."""
        for key, value in self.to_env_vars().items():
            os.environ[key] = value


def load_credentials(path: Optional[Path] = None) -> Credentials:
    """Load credentials from file or environment."""
    if path is None:
        path = DEFAULT_CREDENTIALS_FILE

    # Load from file if exists
    credentials = Credentials.from_yaml(path)

    # Override with environment variables
    if os.environ.get("OPENAI_API_KEY"):
        credentials.openai.api_key = os.environ["OPENAI_API_KEY"]
    if os.environ.get("CLASSIFIER_MODEL"):
        credentials.judge.model = os.environ["CLASSIFIER_MODEL"]

    return credentials


def ensure_directories():
    """Ensure required directories exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
