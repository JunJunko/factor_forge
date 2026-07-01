class FactorForgeError(Exception):
    """Base class for expected platform errors."""


class ContractError(FactorForgeError):
    """Input does not satisfy a versioned platform contract."""


class DataQualityError(FactorForgeError):
    """A blocking data quality rule failed."""


class DSLValidationError(FactorForgeError):
    """A factor formula uses syntax or operators outside the V1 DSL."""


class GateRejected(FactorForgeError):
    """A staged experiment stopped at its configured gate."""


class UnsupportedFactorKindError(ContractError):
    """The factor YAML declares an unsupported ``kind``."""


class FactorCombinationError(ContractError):
    """A factor combination failed validation or execution."""
