# Nix Development Workflows

Nix offers powerful tools for creating reproducible and isolated development environments. This guide explores practical development workflows using Nix on macOS, with a focus on real-world examples.

## Why Use Nix for Development?

Nix provides several advantages for development environments:

- **Reproducibility**: Consistent environments across machines and time
- **Isolation**: Avoid "works on my machine" problems
- **Declarative**: Define environments as code
- **Determinism**: Exact same dependencies every time
- **Cross-platform**: Works on macOS, Linux, and other platforms
- **Version pinning**: Lock dependencies to specific versions
- **Zero conflict**: Multiple projects with different versions of the same dependency

## Getting Started

### Prerequisites

- Nix package manager installed (see [[Installing Nix on macOS]])
- Flakes enabled (recommended, see [[Nix Flakes Introduction]])
- Basic understanding of Nix concepts (see [[What is Nix]])

## Development Shell Approaches

### 1. Quick Temporary Environments

#### Ad-hoc Development Shell

For quick, temporary environments:

```bash
# Python data science environment
nix shell nixpkgs#python3 nixpkgs#python3Packages.pandas nixpkgs#python3Packages.matplotlib

# Node.js development
nix shell nixpkgs#nodejs nixpkgs#yarn nixpkgs#nodePackages.typescript

# Rust development
nix shell nixpkgs#rustc nixpkgs#cargo nixpkgs#rust-analyzer
```

#### One-off Command Execution

Run a command within a temporary environment:

```bash
# Run Python script with dependencies
nix shell nixpkgs#python3 nixpkgs#python3Packages.requests --command python script.py

# Build a Rust project without installing Rust
nix shell nixpkgs#rustc nixpkgs#cargo --command cargo build
```

### 2. Project-specific Development Environments with Flakes

#### Basic Development Shell with Flakes

Create a `flake.nix` in your project:

```nix
{
  description = "My project development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Tools you need
            nodejs
            yarn
            nodePackages.typescript
          ];
          
          # Environment variables
          shellHook = ''
            echo "Node.js development environment activated!"
            export NODE_ENV=development
          '';
        };
      }
    );
}
```

Enter the development environment:

```bash
# In the project directory
nix develop

# Or from anywhere
nix develop /path/to/project
```

Run a command in the environment without entering it:

```bash
nix develop --command yarn install
```

#### Language-specific Examples

##### Python Project

```nix
{
  description = "Python project environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Create a Python with packages
        python-env = pkgs.python3.withPackages (ps: with ps; [
          pandas
          numpy
          matplotlib
          requests
          pytest
          black
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python-env
            poetry
          ];
          
          shellHook = ''
            echo "Python ${python-env.pythonVersion} environment activated!"
            # Optional: Create/activate virtualenv
            if [ ! -d .venv ]; then
              python -m venv .venv
            fi
            source .venv/bin/activate
          '';
        };
      }
    );
}
```

##### Rust Project

```nix
{
  description = "Rust project environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        
        # Specify the Rust version
        rustVersion = pkgs.rust-bin.stable.latest.default;
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust
            rustVersion
            rust-analyzer
            clippy
            
            # Platform-specific dependencies
            pkgconfig
            openssl
          ] ++ lib.optionals stdenv.isDarwin [
            # macOS-specific libraries
            darwin.apple_sdk.frameworks.Security
            darwin.apple_sdk.frameworks.SystemConfiguration
          ];
          
          shellHook = ''
            echo "Rust $(rustc --version) environment activated!"
          '';
        };
      }
    );
}
```

##### Web Development (React/TypeScript)

```nix
{
  description = "React TypeScript project environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            nodejs
            yarn
            nodePackages.typescript
            nodePackages.eslint
            nodePackages.prettier
            nodePackages.typescript-language-server
          ];
          
          shellHook = ''
            echo "Node.js $(node --version) environment activated!"
            export PATH="$PWD/node_modules/.bin:$PATH"
          '';
        };
      }
    );
}
```

### 3. Advanced Development Workflows

#### Multiple Development Environments

Define different environments for various tasks:

```nix
{
  description = "Project with multiple environments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        # Default development environment
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [ nodejs yarn ];
        };
        
        # Environment for frontend work
        devShells.frontend = pkgs.mkShell {
          buildInputs = with pkgs; [
            nodejs
            yarn
            nodePackages.typescript
            nodePackages.prettier
          ];
          
          shellHook = ''
            echo "Frontend development environment activated!"
            cd frontend
          '';
        };
        
        # Environment for backend work
        devShells.backend = pkgs.mkShell {
          buildInputs = with pkgs; [
            go
            golangci-lint
            gotools
          ];
          
          shellHook = ''
            echo "Backend development environment activated!"
            cd backend
          '';
        };
        
        # Environment for database work
        devShells.db = pkgs.mkShell {
          buildInputs = with pkgs; [
            postgresql
            sqlite
            pgcli
          ];
          
          shellHook = ''
            echo "Database tools environment activated!"
          '';
        };
      }
    );
}
```

Enter a specific environment:

```bash
nix develop .#frontend
nix develop .#backend
nix develop .#db
```

#### Integration with direnv

Automatically activate environments when entering directories.

1. Install direnv:
   ```bash
   nix profile install nixpkgs#direnv
   ```

2. Add to your shell config (e.g., `~/.zshrc`):
   ```bash
   eval "$(direnv hook zsh)"
   ```

3. Create `.envrc` in your project:
   ```bash
   use flake
   ```

4. Allow the direnv configuration:
   ```bash
   direnv allow
   ```

Now the environment loads automatically when you enter the directory.

#### For different environments in subdirectories:

```bash
# In frontend/.envrc
use flake ..#frontend

# In backend/.envrc
use flake ..#backend
```

#### Integration with VSCode

Create a `.vscode/settings.json` file:

```json
{
  "nix.enableLanguageServer": true,
  "terminal.integrated.profiles.linux": {
    "nix develop": {
      "path": "nix",
      "args": ["develop", "--command", "zsh"]
    }
  },
  "terminal.integrated.defaultProfile.linux": "nix develop"
}
```

## Practical Examples for macOS

### macOS-specific Development Environment

```nix
{
  description = "macOS-optimized development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Check if running on macOS (either Intel or Apple Silicon)
        isMacOS = pkgs.stdenv.isDarwin;
        # Check if running on Apple Silicon
        isAppleSilicon = isMacOS && pkgs.stdenv.isAarch64;
        
        # Libraries specific to macOS
        macOSLibs = with pkgs.darwin.apple_sdk.frameworks; [
          CoreServices
          CoreFoundation
          Security
          SystemConfiguration
        ];
        
        # Libraries needed only on non-macOS platforms
        linuxLibs = with pkgs; [
          libGL
          libGLU
          libxkbcommon
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
        ];
        
        # Platform-specific libraries
        platformLibs = if isMacOS then macOSLibs else linuxLibs;
        
        # Architecture-specific settings
        archSettings = if isAppleSilicon then {
          # Apple Silicon specific settings
          MACOSX_DEPLOYMENT_TARGET = "12.0";
          # For compatibility with some x86-only packages
          NIX_LDFLAGS = "-L/opt/homebrew/opt/openssl@1.1/lib";
          NIX_CFLAGS_COMPILE = "-I/opt/homebrew/opt/openssl@1.1/include";
        } else {};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Development tools
            cmake
            ninja
            pkg-config
            
            # Libraries
            openssl
            zlib
          ] ++ platformLibs;
          
          # Shell environment variables
          shellHook = ''
            echo "Development environment for ${system} activated!"
            
            ${if isAppleSilicon then ''
              echo "Apple Silicon detected, setting up additional environment variables..."
              export MACOSX_DEPLOYMENT_TARGET=${archSettings.MACOSX_DEPLOYMENT_TARGET}
              export NIX_LDFLAGS="${archSettings.NIX_LDFLAGS}"
              export NIX_CFLAGS_COMPILE="${archSettings.NIX_CFLAGS_COMPILE}"
            '' else ""}
            
            ${if isMacOS then ''
              echo "macOS environment ready!"
            '' else ''
              echo "Linux environment ready!"
            ''}
          '';
        };
      }
    );
}
```

### Cross-compilation for iOS Development

```nix
{
  description = "iOS development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Only proceed if on macOS
        isMacOS = pkgs.stdenv.isDarwin;
        
        # iOS development frameworks
        iosFrameworks = with pkgs.darwin.apple_sdk.frameworks; [
          Foundation
          UIKit
          CoreGraphics
          CoreLocation
          CoreData
          CoreText
          Security
        ];
      in {
        devShells = {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              xcbuild # Xcode build tools
              cocoapods
              darwin.ios-deploy
            ] ++ (if isMacOS then iosFrameworks else []);
            
            shellHook = ''
              if [[ "${system}" != *darwin* ]]; then
                echo "Warning: iOS development typically requires macOS."
                echo "Some functionality may be limited on this platform."
              else
                echo "iOS development environment activated!"
                export DEVELOPER_DIR=$(xcode-select -p)
                echo "Using Xcode from: $DEVELOPER_DIR"
              fi
            '';
          };
        };
      }
    );
}
```

### Full-stack Development with Local Services

```nix
{
  description = "Full-stack development environment with local services";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # macOS-specific libraries
        darwinLibs = with pkgs.darwin.apple_sdk.frameworks; [
          Security
          CoreFoundation
          CoreServices
        ];
        
        platformSpecificLibs = if pkgs.stdenv.isDarwin then darwinLibs else [];
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Frontend tools
            nodejs
            nodePackages.typescript
            yarn
            
            # Backend tools
            go
            
            # Database
            postgresql_15
            redis
            
            # Tools
            docker
            docker-compose
            jq
            xdg-utils
          ] ++ platformSpecificLibs;
          
          shellHook = ''
            # Welcome message
            echo "🚀 Full-stack development environment activated!"
            
            # Set up local Postgres
            export PGDATA=$PWD/.postgres
            export PGHOST=$PGDATA
            export PGPORT=5432
            export PGDATABASE=devdb
            export DATABASE_URL="postgresql://localhost:$PGPORT/$PGDATABASE"
            
            if [ ! -d $PGDATA ]; then
              echo "Initializing PostgreSQL database in $PGDATA..."
              initdb --auth=trust --no-locale
              # Start the database
              pg_ctl start -l $PGDATA/postgres.log
              # Create the database
              createdb $PGDATABASE
            else
              # Start the database if it's not running
              pg_ctl status || pg_ctl start -l $PGDATA/postgres.log
            fi
            
            # Set up Redis if needed
            export REDIS_PORT=6379
            
            # Add node_modules/.bin to PATH
            export PATH="$PWD/node_modules/.bin:$PATH"
            
            # Set development environment variables
            export NODE_ENV=development
            export GO111MODULE=on
            
            # Helper functions
            start_services() {
              echo "Starting all services..."
              # Start Redis in the background
              redis-server --port $REDIS_PORT &
              echo "Redis started on port $REDIS_PORT"
            }
            
            stop_services() {
              echo "Stopping all services..."
              # Stop Redis
              pkill redis-server || true
              # Stop PostgreSQL
              pg_ctl stop || true
            }
            
            # Clean up when shell exits
            trap "stop_services" EXIT
            
            # Print help info
            echo ""
            echo "📋 Available commands:"
            echo "  • start_services - Start Redis and ensure PostgreSQL is running"
            echo "  • stop_services - Stop all background services"
            echo ""
            echo "💾 Database information:"
            echo "  • PostgreSQL running on port $PGPORT"
            echo "  • Database name: $PGDATABASE"
            echo "  • Connection URL: $DATABASE_URL"
            echo ""
            echo "Happy coding! 🎉"
          '';
        };
      }
    );
}
```

## Testing and CI/CD

### Setting Up a Test Environment

```nix
{
  description = "Project with testing environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        # Development environment
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            nodejs
            yarn
            nodePackages.typescript
          ];
        };
        
        # Test environment
        devShells.test = pkgs.mkShell {
          buildInputs = with pkgs; [
            nodejs
            yarn
            nodePackages.typescript
            nodePackages.jest
            chromium
            firefox
          ];
          
          shellHook = ''
            echo "Test environment activated!"
            export CHROME_BIN=${pkgs.chromium}/bin/chromium
            export FIREFOX_BIN=${pkgs.firefox}/bin/firefox
          '';
        };
        
        # Creates a package for CI pipelines to use
        packages.default = pkgs.stdenv.mkDerivation {
          name = "my-project";
          src = ./.;
          buildInputs = with pkgs; [
            nodejs
            yarn
          ];
          buildPhase = ''
            export HOME=$(mktemp -d)
            yarn install --frozen-lockfile
            yarn build
          '';
          installPhase = ''
            mkdir -p $out
            cp -r dist $out/
          '';
        };
      }
    );
}
```

### GitHub Actions Integration

Create `.github/workflows/build.yml`:

```yaml
name: Build and Test

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Nix
        uses: cachix/install-nix-action@v20
        with:
          extra_nix_config: |
            experimental-features = nix-command flakes
      
      - name: Cachix
        uses: cachix/cachix-action@v12
        with:
          name: your-cache-name
          authToken: '${{ secrets.CACHIX_AUTH_TOKEN }}'
      
      - name: Build
        run: nix build
      
      - name: Test
        run: nix develop .#test --command yarn test
```

## Best Practices

### Structure and Organization

1. **Separate configuration by purpose**:
   - Put common settings in a base configuration
   - Use separate files for language-specific settings
   - Create dedicated environments for specific tasks

2. **Use a consistent directory structure**:
   ```
   project/
   ├── flake.nix        # Main flake file
   ├── flake.lock       # Lock file (auto-generated)
   ├── nix/             # Nix-related files
   │   ├── common.nix   # Common development settings
   │   ├── python.nix   # Python-specific settings
   │   └── node.nix     # Node.js-specific settings
   ├── .envrc           # direnv configuration
   └── .github/         # GitHub Actions configuration
   ```

3. **Document your environments**:
   - Add comments in your Nix files
   - Create a README with environment setup instructions
   - Include examples of common tasks

### Performance Optimization

1. **Use binary caches**:
   - Set up Cachix for your project
   - Configure substituters for faster builds

2. **Minimize environment size**:
   - Only include necessary packages
   - Use `nativeBuildInputs` for build-time dependencies

3. **Offline development**:
   - Pin dependencies in `flake.lock`
   - Use `--offline` flag when possible

### Collaboration

1. **Share development environments**:
   - Include `flake.nix` and `flake.lock` in version control
   - Document any system-specific requirements

2. **Standardize tooling**:
   - Define linters, formatters, and editors in your environment
   - Integrate with `.editorconfig` and similar tools

3. **Onboarding automation**:
   - Create scripts for first-time setup
   - Provide self-documenting shell hooks

## Relation to Other Concepts

- **[[What is Nix]]**: Foundation for understanding Nix development
- **[[Basic Nix Commands]]**: Essential commands for development workflows
- **[[Nix Flakes Introduction]]**: Core technology for reproducible environments
- **[[Working with Nix Packages]]**: How to add and manage packages
- **[[Home Manager Configuration]]**: User environment management

## Next Steps

→ Learn about [[Nix Package Development]] for creating your own packages
→ Set up [[Nix Darwin Configuration]] for system-wide macOS settings
→ Explore [[Containerization with Nix]] for Docker integration

---
Tags: #nix #development #flakes #devops #macos #workflows 