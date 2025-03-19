# Nix Darwin Configuration

`nix-darwin` is a powerful tool for managing macOS system configuration using the Nix package manager. This guide explains how to use `nix-darwin` to declaratively configure your macOS system.

## What is nix-darwin?

`nix-darwin` is a collection of Nix modules that allows you to:
- Declaratively manage macOS system settings
- Configure system packages, services, and preferences
- Version control your entire system configuration
- Apply system changes reliably and reproducibly

It serves a similar purpose to NixOS, but for macOS systems, allowing you to manage your Apple environment with the same declarative approach.

## Installation

### Prerequisites

- Nix package manager installed (see [[Installing Nix on macOS]])
- Flakes enabled (recommended, see [[Nix Flakes Introduction]])

### Installation Methods

#### Using Flakes (Recommended)

1. Create a directory for your configuration:
   ```bash
   mkdir -p ~/.config/darwin
   cd ~/.config/darwin
   ```

2. Create a basic `flake.nix` file:
   ```nix
   {
     description = "Darwin configuration";
   
     inputs = {
       nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
       darwin.url = "github:lnl7/nix-darwin";
       darwin.inputs.nixpkgs.follows = "nixpkgs";
     };
   
     outputs = { self, darwin, nixpkgs }: {
       darwinConfigurations.your-hostname = darwin.lib.darwinSystem {
         system = "aarch64-darwin"; # or "x86_64-darwin" for Intel
         modules = [ ./configuration.nix ];
       };
     };
   }
   ```

3. Create a basic `configuration.nix` file:
   ```nix
   { pkgs, ... }: {
     # System-wide packages
     environment.systemPackages = with pkgs; [
       vim
       git
     ];
     
     # Use touch ID for sudo auth
     security.pam.enableSudoTouchIdAuth = true;
     
     # Auto upgrade nix package and the daemon service
     services.nix-daemon.enable = true;
     
     # Allow unfree packages
     nixpkgs.config.allowUnfree = true;
     
     # Enable experimental nix features
     nix.settings.experimental-features = [ "nix-command" "flakes" ];
     
     # Create /etc/zshrc that loads the nix-darwin environment
     programs.zsh.enable = true;
     
     # Set hostname
     networking.hostName = "your-hostname";
     
     # System defaults
     system.defaults.dock.autohide = true;
     system.defaults.finder.AppleShowAllExtensions = true;
     
     # Fonts
     fonts.fontDir.enable = true;
     
     # System version
     system.stateVersion = 4;
   }
   ```

4. Build and switch to the configuration:
   ```bash
   nix run nix-darwin -- switch --flake .
   ```

#### Using Traditional Approach

1. Install nix-darwin:
   ```bash
   nix-build https://github.com/LnL7/nix-darwin/archive/master.tar.gz -A installer
   ./result/bin/darwin-installer
   ```

2. Edit the generated configuration at `~/.nixpkgs/darwin-configuration.nix`

3. Apply changes:
   ```bash
   darwin-rebuild switch
   ```

## Core Configuration Areas

### Environment and Packages

Configure system-wide packages:

```nix
environment.systemPackages = with pkgs; [
  coreutils
  curl
  wget
  git
  neovim
];

# Set environment variables
environment.variables = {
  EDITOR = "vim";
};

# Add shells
environment.shells = with pkgs; [
  bash
  zsh
  fish
];
```

### System Defaults

Configure macOS system preferences:

```nix
system.defaults = {
  dock = {
    autohide = true;
    orientation = "left";
    showhidden = true;
    mru-spaces = false;
  };
  
  finder = {
    AppleShowAllExtensions = true;
    QuitMenuItem = true;
    FXEnableExtensionChangeWarning = false;
  };
  
  trackpad = {
    Clicking = true;
    TrackpadRightClick = true;
  };
  
  # Global macOS preferences
  NSGlobalDomain = {
    AppleShowAllExtensions = true;
    AppleKeyboardUIMode = 3;
    ApplePressAndHoldEnabled = false;
    InitialKeyRepeat = 10;
    KeyRepeat = 1;
    NSAutomaticCapitalizationEnabled = false;
    NSAutomaticDashSubstitutionEnabled = false;
    NSAutomaticPeriodSubstitutionEnabled = false;
    NSAutomaticQuoteSubstitutionEnabled = false;
    NSAutomaticSpellingCorrectionEnabled = false;
    "com.apple.swipescrolldirection" = false;
    "com.apple.keyboard.fnState" = true;
  };
};

# Keyboard
system.keyboard = {
  enableKeyMapping = true;
  remapCapsLockToEscape = true;
};
```

### User Configuration

Configure user accounts:

```nix
users.users.yourusername = {
  name = "yourusername";
  home = "/Users/yourusername";
  shell = pkgs.zsh;
};
```

### Programs and Services

Configure built-in programs:

```nix
# Shell configuration
programs.zsh = {
  enable = true;
  promptInit = ''
    # Custom prompt setup
  '';
};

programs.fish.enable = true;

# Homebrew integration (optional)
homebrew = {
  enable = true;
  onActivation.autoUpdate = true;
  global.brewfile = true;
  
  taps = [
    "homebrew/cask-fonts"
    "homebrew/services"
  ];
  
  brews = [
    "mas" # Mac App Store CLI
  ];
  
  casks = [
    "firefox"
    "visual-studio-code"
    "iterm2"
  ];
  
  masApps = {
    "Xcode" = 497799835;
  };
};

# Enable services
services = {
  nix-daemon.enable = true;
  
  yabai = {
    enable = true;
    package = pkgs.yabai;
    enableScriptingAddition = true;
    config = {
      layout = "bsp";
      window_placement = "second_child";
      # other yabai settings
    };
  };
  
  skhd = {
    enable = true;
    package = pkgs.skhd;
    skhdConfig = ''
      # skhd configuration
      alt - return : open -n /Applications/iTerm.app
    '';
  };
};
```

### Nix Configuration

Configure Nix behavior:

```nix
nix = {
  package = pkgs.nix;
  
  settings = {
    experimental-features = [ "nix-command" "flakes" ];
    auto-optimise-store = true;
    max-jobs = "auto";
    substituters = [
      "https://cache.nixos.org"
      "https://nix-community.cachix.org"
    ];
    trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };
  
  gc = {
    automatic = true;
    interval.Day = 7;
    options = "--delete-older-than 14d";
  };
};
```

### Fonts

Configure system fonts:

```nix
fonts = {
  fontDir.enable = true;
  fonts = with pkgs; [
    jetbrains-mono
    (nerdfonts.override { fonts = [ "JetBrainsMono" "FiraCode" ]; })
    fira-code
    font-awesome
  ];
};
```

### Apple Silicon Specific Configuration

Special considerations for Apple Silicon (M1/M2/M3) Macs:

```nix
# Make sure to specify the correct architecture
darwinConfigurations.your-hostname = darwin.lib.darwinSystem {
  system = "aarch64-darwin";
  # ...
};

# Enable Rosetta 2 for Intel app compatibility
system.rosetta.enable = true;

# Use architecture-specific packages
environment.systemPackages = with pkgs; [
  # For Apple Silicon-specific packages:
  (if pkgs.stdenv.isDarwin && pkgs.stdenv.isAarch64 
    then apple-silicon-package 
    else intel-package)
];
```

## Advanced Usage

### Directory Structure

For larger configurations, consider this directory structure:

```
~/.config/darwin/
├── flake.nix
├── flake.lock
├── configuration.nix
├── modules/
│   ├── system.nix
│   ├── homebrew.nix
│   ├── apps.nix
│   └── development.nix
└── hosts/
    ├── work-macbook.nix
    └── personal-mac-mini.nix
```

### Multi-Host Configuration

Configure multiple Macs with the same codebase:

```nix
{
  description = "My darwin configurations";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    darwin.url = "github:lnl7/nix-darwin";
    darwin.inputs.nixpkgs.follows = "nixpkgs";
    home-manager.url = "github:nix-community/home-manager";
    home-manager.inputs.nixpkgs.follows = "nixpkgs";
  };
  
  outputs = { self, darwin, nixpkgs, home-manager }: {
    darwinConfigurations = {
      # MacBook Pro configuration
      "macbook-pro" = darwin.lib.darwinSystem {
        system = "aarch64-darwin";
        modules = [
          ./configuration.nix
          ./hosts/macbook-pro.nix
          home-manager.darwinModules.home-manager
          {
            home-manager.useGlobalPkgs = true;
            home-manager.useUserPackages = true;
            home-manager.users.yourusername = import ./home.nix;
          }
        ];
      };
      
      # Mac Mini configuration
      "mac-mini" = darwin.lib.darwinSystem {
        system = "aarch64-darwin";
        modules = [
          ./configuration.nix
          ./hosts/mac-mini.nix
          home-manager.darwinModules.home-manager
          {
            home-manager.useGlobalPkgs = true;
            home-manager.useUserPackages = true;
            home-manager.users.yourusername = import ./home.nix;
          }
        ];
      };
    };
  };
}
```

### Integration with Home Manager

Combine nix-darwin with home-manager for complete system and user configuration:

```nix
# In flake.nix
inputs = {
  nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  darwin.url = "github:lnl7/nix-darwin";
  darwin.inputs.nixpkgs.follows = "nixpkgs";
  home-manager.url = "github:nix-community/home-manager";
  home-manager.inputs.nixpkgs.follows = "nixpkgs";
};

outputs = { self, darwin, nixpkgs, home-manager }: {
  darwinConfigurations.your-hostname = darwin.lib.darwinSystem {
    system = "aarch64-darwin";
    modules = [
      ./configuration.nix
      home-manager.darwinModules.home-manager
      {
        home-manager.useGlobalPkgs = true;
        home-manager.useUserPackages = true;
        home-manager.users.yourusername = import ./home.nix;
      }
    ];
  };
};
```

```nix
# In home.nix
{ config, pkgs, ... }: {
  home.packages = with pkgs; [
    ripgrep
    fd
    bat
  ];
  
  programs.git = {
    enable = true;
    userName = "Your Name";
    userEmail = "your.email@example.com";
  };
  
  # ... other home-manager configurations
  
  # This value determines the Home Manager release that your
  # configuration is compatible with.
  home.stateVersion = "23.11";
}
```

## Daily Workflow

### Applying Changes

After modifying your configuration:

**With flakes:**
```bash
darwin-rebuild switch --flake ~/.config/darwin
```

**Without flakes:**
```bash
darwin-rebuild switch
```

### Testing Changes Before Applying

**With flakes:**
```bash
darwin-rebuild check --flake ~/.config/darwin
```

**Without flakes:**
```bash
darwin-rebuild check
```

### Building Without Applying

**With flakes:**
```bash
darwin-rebuild build --flake ~/.config/darwin
```

**Without flakes:**
```bash
darwin-rebuild build
```

## Troubleshooting

### Common Issues

1. **"permission denied" errors**:
   - Ensure you have appropriate permissions for the files and directories

2. **Build failures with specific packages**:
   - Check if the package supports macOS
   - Look for architecture-specific issues (Intel vs. Apple Silicon)

3. **Conflicting files during installation**:
   - Back up and remove conflicting files, then retry

4. **sudo requiring password despite TouchID config**:
   - Check that `security.pam.enableSudoTouchIdAuth = true;` is set
   - Ensure `/etc/pam.d/sudo` has the appropriate modifications

5. **Homebrew integration issues**:
   - Ensure Homebrew is installed
   - Check that paths are correctly set

### Debugging Tips

1. Add verbose output to see more details:
   ```bash
   darwin-rebuild switch -v
   ```

2. Check the Nix store for logs:
   ```bash
   nix log /nix/store/hash-darwin-system
   ```

3. Inspect system activation:
   ```bash
   ls -la /run/current-system/
   cat /run/current-system/darwin-version
   ```

## Relation to Other Concepts

- **[[What is Nix]]**: Foundation for understanding the Nix ecosystem
- **[[Installing Nix on macOS]]**: Prerequisites for using nix-darwin
- **[[Nix Flakes Introduction]]**: Modern approach to Nix configurations
- **[[Basic Nix Commands]]**: Commands used with nix-darwin
- **[[Working with Nix Packages]]**: Managing packages in your configuration

## Next Steps

→ Learn about [[Home Manager Configuration]] for user-specific settings
→ Explore [[Working with Nix Packages]] for package management
→ Set up [[Nix Development Workflows]] for project environments

---
Tags: #nix #nix-darwin #macos #system-configuration #flakes 