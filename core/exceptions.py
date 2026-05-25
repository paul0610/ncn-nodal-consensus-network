"""NCN custom exceptions."""


class NCNError(Exception):
    """Base exception for all NCN errors."""


class ConfigError(NCNError):
    """Base for configuration errors."""


class ConfigFileNotFoundError(ConfigError):
    """Config file does not exist."""


class ConfigValidationError(ConfigError):
    """Config file failed Pydantic validation."""


class GraphError(NCNError):
    """Base for graph database errors."""


class GraphConnectionError(GraphError):
    """Cannot connect to the graph database."""


class GraphNodeNotFoundError(GraphError):
    """Requested node does not exist."""


class GraphSchemaError(GraphError):
    """Schema creation or migration failed."""


class GraphWriteError(GraphError):
    """Write operation failed."""


class ProviderError(NCNError):
    """Base for model provider errors."""


class ConsensusError(NCNError):
    """Base for consensus engine errors."""


class TermsNotAcceptedError(NCNError):
    """User must accept provider terms before using bootstrap."""


class AgentError(NCNError):
    """Base for agent errors."""


class AgentDisabledError(AgentError):
    """Agent features are not enabled in config."""


class DangerousCommandError(AgentError):
    """Command matched a dangerous pattern and was blocked."""


class CommandTimeoutError(AgentError):
    """Command exceeded the configured timeout."""


class SkillNotFoundError(AgentError):
    """Requested skill does not exist."""
