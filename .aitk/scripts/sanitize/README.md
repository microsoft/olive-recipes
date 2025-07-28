# Sanitize Module Refactoring

This directory contains the refactored sanitize functionality, split into multiple modules for better maintainability and organization.

## File Structure

```
sanitize/
├── __init__.py              # Package initialization and exports
├── constants.py             # All constants, enums, and static values
├── utils.py                 # Utility functions and GlobalVars
├── base.py                  # Base model classes
├── model_info.py            # ModelInfo and ModelList classes
├── parameters.py            # Parameter-related classes
├── project_config.py        # Project configuration classes
├── model_parameter.py       # ModelParameter and related classes
├── copy_config.py           # Copy configuration functionality
└── file_validation.py       # File validation functions
```

## Main Files

- `sanitize.py` - Main entry point (maintains compatibility)
- `sanitize_main.py` - Contains the main logic function
- `sanitize_old.py` - Backup of the original monolithic file

## Module Descriptions

### constants.py
Contains all enums and constant values:
- `IconEnum`, `ArchitectureEnum`, `ModelStatusEnum`
- `ParameterTypeEnum`, `ParameterDisplayTypeEnum`, etc.
- `EPNames`, `OlivePassNames`, `OlivePropertyNames`
- Path and import constants

### utils.py
Utility functions and shared state:
- `GlobalVars` - Global state management with direct initialization
  - All runtime mappings are initialized at module import time
  - `epToName` - EP to display name mappings (from constants)
  - `runtimeToEp` - Runtime enum to EP string mappings
- `printProcess`, `printInfo`, `printError`, `printWarning` - Logging functions
- `open_ex` - File I/O context manager
- `checkPath` - Path validation utility

#### Initialization Design

### base.py
Base classes for all model classes:
- `BaseModelClass` - Base class with file I/O capabilities

### model_info.py
Model information management:
- `ModelInfo` - Individual model information
- `ModelList` - Collection of models with validation

### parameters.py
Parameter configuration:
- `Parameter` - Parameter definition and validation
- `ParameterCheck`, `ParameterAction` - Parameter validation and actions
- `readCheckParameterTemplate` - Template reading function

### project_config.py
Project configuration management:
- `WorkflowItem` - Individual workflow configuration
- `ModelInfoProject` - Project-specific model info
- `ModelProjectConfig` - Complete project configuration

### model_parameter.py
Model parameter configuration:
- `ModelParameter` - Main model parameter class
- `Section` - Configuration sections
- `RuntimeOverwrite`, `DebugInfo`, `ADMNPUConfig` - Supporting classes

### copy_config.py
File copy and replacement functionality:
- `Copy`, `Replacement` - Copy operations
- `CopyConfig` - Copy configuration management

### file_validation.py
File validation and checking:
- `check_case` - Path case validation
- `process_gitignore` - Git ignore file processing
- `readCheckOliveConfig` - Olive configuration validation
- `readCheckIpynb` - Jupyter notebook validation

## Usage

The new structure maintains full compatibility with the original script. You can still run:

```bash
python sanitize.py
python sanitize.py -v
python sanitize.py --olive /path/to/olive/repo
```

## Benefits of Refactoring

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Individual modules can be tested in isolation
3. **Reusability**: Components can be imported and used separately
4. **Readability**: Smaller, focused files are easier to understand
5. **Type Safety**: Better type hints and error checking
6. **Extensibility**: New functionality can be added without modifying existing code

## Migration Notes

- All functionality from the original `sanitize.py` is preserved
- Import statements may need to be updated if you were importing specific classes
- The main entry point remains the same for backward compatibility
- Error handling and logging behavior is unchanged
