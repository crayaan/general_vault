# Setting Up nix-darwin

This guide walks through the process of setting up and configuring nix-darwin on macOS, providing a declarative approach to managing your macOS system configuration.

## Overview

`nix-darwin` is to macOS what NixOS is to Linux—a way to declaratively configure your operating system using the Nix language. It allows you to manage system settings, packages, and services in a reproducible way.

Key benefits include:
- Declarative macOS configuration
- Version-controlled system setup
- Reproducible environments across multiple Macs
- Atomic updates and rollbacks

## Prerequisites

- **Nix installed** on your macOS system (see [[Installing Nix on macOS]])
- **Flakes enabled** for modern workflow (see [[Nix Flakes Introduction]])
- **Administrator privileges** on your Mac
- **Command-line familiarity**

## Basic Installation

### Method 1: Using Flakes (Recommended)

1. **Create a configuration directory**:
   ```bash
   mkdir -p ~/.config/darwin
   cd ~/.config/darwin
   ```

2. **Create a basic flake.nix file**:
   ```nix
   {
     description = "Darwin system configuration";
     
     inputs = {
       nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
       darwin.url = "github:lnl7/nix-darwin";
       darwin.inputs.nixpkgs.follows = "nixpkgs";
     };
     
     outputs = { self, nixpkgs, darwin }: {
       darwinConfigurations."your-hostname" = darwin.lib.darwinSystem {
         system = "aarch64-darwin"; # or "x86_64-darwin" for Intel Macs
         modules = [ ./configuration.nix ];
       };
     };
   }
   ```
   
   > ℹ️ Replace "your-hostname" with your Mac's hostname. You can check your hostname by running `scutil --get LocalHostName` in Terminal.

3. **Create a basic configuration.nix file**:
   ```nix
   { pkgs, ... }: {
     # Ensure nix-daemon is running
     services.nix-daemon.enable = true;
     
     # Auto upgrade nix package and the daemon service
     nix.settings.auto-optimise-store = true;
     
     # Allow unfree packages
     nixpkgs.config.allowUnfree = true;
     
     # Enable experimental nix command and flakes
     nix.settings.experimental-features = [ "nix-command" "flakes" ];
     
     # Create /etc/zshrc that loads the nix-darwin environment
     programs.zsh.enable = true;
     
     # Basic system packages
     environment.systemPackages = with pkgs; [
       vim
       git
       curl
     ];
     
     # Set Apple defaults
     system.defaults.finder.AppleShowAllExtensions = true;
     system.defaults.dock.autohide = true;
     
     # Use Touch ID for sudo authentication
     security.pam.enableSudoTouchIdAuth = true;
     
     # System version
     system.stateVersion = 4;
   }
   ```

4. **Bootstrap nix-darwin**:
   ```bash
   nix run nix-darwin -- switch --flake .
   ```
   
   This command may prompt you about files that need to be backed up or deleted. You'll typically be able to say yes to these prompts for a fresh installation.

5. **Restart your terminal session** to ensure all changes take effect.

6. **Verify the installation**:
   ```bash
   darwin-rebuild --version
   ```

### Method 2: Using Traditional Installation (Non-Flakes)

1. **Install nix-darwin**:
   ```bash
   nix-build https://github.com/LnL7/nix-darwin/archive/master.tar.gz -A installer
   ./result/bin/darwin-installer
   ```

2. **Edit the generated configuration file**:
   ```bash
   vim ~/.nixpkgs/darwin-configuration.nix
   ```

3. **Apply your changes**:
   ```bash
   darwin-rebuild switch
   ```

## Integration with Home Manager

For a complete system configuration, combine nix-darwin with Home Manager to manage both system and user-level configurations:

1. **Update your flake.nix to include Home Manager**:
   ```nix
   {
     description = "Darwin system configuration";
     
     inputs = {
       nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
       darwin.url = "github:lnl7/nix-darwin";
       darwin.inputs.nixpkgs.follows = "nixpkgs";
       home-manager.url = "github:nix-community/home-manager";
       home-manager.inputs.nixpkgs.follows = "nixpkgs";
     };
     
     outputs = { self, nixpkgs, darwin, home-manager }: {
       darwinConfigurations."your-hostname" = darwin.lib.darwinSystem {
         system = "aarch64-darwin"; # or "x86_64-darwin" for Intel Macs
         modules = [
           ./configuration.nix
           
           # Add home-manager module
           home-manager.darwinModules.home-manager {
             home-manager = {
               useGlobalPkgs = true;
               useUserPackages = true;
               users.yourusername = import ./home.nix;
             };
           }
         ];
       };
     };
   }
   ```

2. **Create a home.nix file** for user-specific configurations:
   ```nix
   { pkgs, ... }: {
     home.packages = with pkgs; [
       ripgrep
       fd
       jq
     ];
     
     programs.git = {
       enable = true;
       userName = "Your Name";
       userEmail = "your.email@example.com";
     };
     
     programs.zsh = {
       enable = true;
       enableAutosuggestions = true;
       enableSyntaxHighlighting = true;
     };
     
     # This value determines the Home Manager release that your
     # configuration is compatible with
     home.stateVersion = "23.11";
   }
   ```

3. **Apply the combined configuration**:
   ```bash
   darwin-rebuild switch --flake .
   ```

## Advanced Configuration Examples

### System-Wide Defaults

Here's a configuration with more macOS system defaults:

```nix
# In configuration.nix
{ pkgs, ... }: {
  # ...existing config...
  
  # Dock settings
  system.defaults.dock = {
    autohide = true;
    autohide-delay = 0.0;
    autohide-time-modifier = 0.2;
    expose-animation-duration = 0.1;
    mru-spaces = false;
    orientation = "bottom";
    show-recents = false;
    static-only = true;
    tilesize = 48;
  };
  
  # Finder settings
  system.defaults.finder = {
    AppleShowAllExtensions = true;
    FXEnableExtensionChangeWarning = false;
    QuitMenuItem = true;
    ShowPathbar = true;
    ShowStatusBar = true;
  };
  
  # Trackpad settings
  system.defaults.trackpad = {
    Clicking = true;
    TrackpadRightClick = true;
  };
  
  # Global settings
  system.defaults.NSGlobalDomain = {
    AppleInterfaceStyle = "Dark"; # Dark mode
    AppleKeyboardUIMode = 3; # Full keyboard access
    ApplePressAndHoldEnabled = false; # Disable press-and-hold for keys
    InitialKeyRepeat = 15; # Key repeat initial delay
    KeyRepeat = 2; # Key repeat interval
    NSAutomaticCapitalizationEnabled = false;
    NSAutomaticDashSubstitutionEnabled = false;
    NSAutomaticPeriodSubstitutionEnabled = false;
    NSAutomaticQuoteSubstitutionEnabled = false;
    NSAutomaticSpellingCorrectionEnabled = false;
    "com.apple.keyboard.fnState" = true; # Use F1, F2 as standard keys
    "com.apple.sound.beep.feedback" = 0; # Disable beep sound
    "com.apple.swipescrolldirection" = false; # "Natural" scrolling
  };
}
```

### Homebrew Integration

Manage Homebrew packages and casks alongside your nix-darwin configuration:

```nix
# In configuration.nix
{ pkgs, ... }: {
  # ...existing config...
  
  # Enable Homebrew integration
  homebrew = {
    enable = true;
    onActivation = {
      autoUpdate = true;
      upgrade = true;
      cleanup = "zap"; # Remove outdated versions and their cached downloads
    };
    
    # Homebrew taps
    taps = [
      "homebrew/core"
      "homebrew/cask"
      "homebrew/cask-fonts"
    ];
    
    # Homebrew packages (CLI tools)
    brews = [
      "mas" # Mac App Store CLI
      "imagemagick"
    ];
    
    # Homebrew casks (apps)
    casks = [
      "firefox"
      "rectangle"
      "1password"
      "alfred"
      "visual-studio-code"
      "font-fira-code"
    ];
    
    # Mac App Store apps
    masApps = {
      "Keynote" = 409183694;
      "Numbers" = 409203825;
      "Pages" = 409201541;
    };
  };
}
```

### Multiple Machine Configuration

For managing multiple Macs with different configurations:

1. **Create a directory structure**:
   ```
   ~/.config/darwin/
   ├── flake.nix
   ├── flake.lock
   ├── common.nix                # Shared configuration
   ├── machines/
   │   ├── macbook-pro.nix       # MacBook Pro specific
   │   └── mac-mini.nix          # Mac Mini specific
   ├── modules/
   │   ├── core.nix              # Core system settings
   │   ├── homebrew.nix          # Homebrew configuration
   │   └── development.nix       # Development tools
   └── home/
       ├── common.nix            # Common home-manager config
       ├── work.nix              # Work profile
       └── personal.nix          # Personal profile
   ```

2. **Create a modular flake.nix**:
   ```nix
   {
     description = "Multi-machine Darwin configuration";
     
     inputs = {
       nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
       darwin.url = "github:lnl7/nix-darwin";
       darwin.inputs.nixpkgs.follows = "nixpkgs";
       home-manager.url = "github:nix-community/home-manager";
       home-manager.inputs.nixpkgs.follows = "nixpkgs";
     };
     
     outputs = { self, nixpkgs, darwin, home-manager }:
     let
       # Shared configuration across machines
       sharedModules = [
         ./common.nix
         ./modules/core.nix
         
         # Home Manager module
         home-manager.darwinModules.home-manager
       ];
     in {
       darwinConfigurations = {
         # MacBook Pro configuration
         "macbook-pro" = darwin.lib.darwinSystem {
           system = "aarch64-darwin";
           modules = sharedModules ++ [
             ./machines/macbook-pro.nix
             {
               # MacBook-specific home-manager config
               home-manager.useGlobalPkgs = true;
               home-manager.useUserPackages = true;
               home-manager.users.yourusername = { imports = [ ./home/common.nix ./home/work.nix ]; };
             }
           ];
         };
         
         # Mac Mini configuration
         "mac-mini" = darwin.lib.darwinSystem {
           system = "aarch64-darwin";
           modules = sharedModules ++ [
             ./machines/mac-mini.nix
             {
               # Mac Mini-specific home-manager config
               home-manager.useGlobalPkgs = true;
               home-manager.useUserPackages = true;
               home-manager.users.yourusername = { imports = [ ./home/common.nix ./home/personal.nix ]; };
             }
           ];
         };
       };
     };
   }
   ```

3. **Apply configuration to a specific machine**:
   ```bash
   darwin-rebuild switch --flake .#macbook-pro
   ```

### Apple Silicon-specific Configuration

Configure settings specifically for Apple Silicon Macs:

```nix
# In configuration.nix
{ pkgs, lib, ... }: {
  # ...existing config...
  
  # Architecture-specific configuration
  nix.settings = {
    # Use both architectures on Apple Silicon
    extra-platforms = lib.mkIf (pkgs.stdenv.isDarwin && pkgs.stdenv.isAarch64) 
      [ "x86_64-darwin" "aarch64-darwin" ];
  };
  
  # Enable Rosetta for Intel app compatibility
  system.rosetta.enable = true;
  
  # Custom packages based on architecture
  environment.systemPackages = with pkgs; [
    # Common packages
    git
    vim
    curl
  ] ++ (if (pkgs.stdenv.isDarwin && pkgs.stdenv.isAarch64) then [
    # Apple Silicon specific
    (pkgs.writeShellScriptBin "arch-check" ''
      echo "Running on $(uname -m)"
      echo "Homebrew location: $(which brew)"
    '')
  ] else [
    # Intel specific
  ]);
}
```

## Common Services Setup

### Yabai Window Manager

Configure the yabai tiling window manager:

```nix
# In configuration.nix
{ pkgs, ... }: {
  # ...existing config...
  
  # Enable yabai service
  services.yabai = {
    enable = true;
    package = pkgs.yabai;
    enableScriptingAddition = true; # Note: requires SIP to be partially disabled
    config = {
      focus_follows_mouse = "autoraise";
      mouse_follows_focus = "off";
      window_placement = "second_child";
      window_opacity = "off";
      window_shadow = "float";
      window_border = "on";
      window_border_width = 4;
      active_window_border_color = "0xff5c7e81";
      normal_window_border_color = "0xff505050";
      layout = "bsp";
      top_padding = 5;
      bottom_padding = 5;
      left_padding = 5;
      right_padding = 5;
      window_gap = 5;
    };
    extraConfig = ''
      # Custom rules
      yabai -m rule --add app="^System Preferences$" manage=off
      yabai -m rule --add app="^Calculator$" manage=off
      yabai -m rule --add app="^Finder$" manage=off
    '';
  };
  
  # Configure skhd (hotkey daemon) to work with yabai
  services.skhd = {
    enable = true;
    package = pkgs.skhd;
    skhdConfig = ''
      # focus window
      alt - h : yabai -m window --focus west
      alt - j : yabai -m window --focus south
      alt - k : yabai -m window --focus north
      alt - l : yabai -m window --focus east

      # swap managed window
      shift + alt - h : yabai -m window --swap west
      shift + alt - j : yabai -m window --swap south
      shift + alt - k : yabai -m window --swap north
      shift + alt - l : yabai -m window --swap east

      # toggle window fullscreen zoom
      alt - f : yabai -m window --toggle zoom-fullscreen
    '';
  };
}
```

### Custom Launch Agents

Add custom launchd services:

```nix
# In configuration.nix
{ pkgs, ... }: {
  # ...existing config...
  
  # Define a custom launchd service
  launchd.user.agents = {
    "filesystem-watch" = {
      serviceConfig = {
        ProgramArguments = [
          "${pkgs.bash}/bin/bash"
          "-c"
          "/Users/yourusername/scripts/watch-filesystem.sh"
        ];
        RunAtLoad = true;
        KeepAlive = true;
        StandardOutPath = "/tmp/filesystem-watch.log";
        StandardErrorPath = "/tmp/filesystem-watch-error.log";
      };
    };
  };
}
```

## Daily Usage and Maintenance

### Applying Configuration Changes

After making changes to your configuration:

```bash
# With flakes
darwin-rebuild switch --flake .

# Without flakes
darwin-rebuild switch
```

### Updating Your System

Keep your system up to date:

```bash
# Update flake inputs
nix flake update

# Apply the updated configuration
darwin-rebuild switch --flake .
```

### Rolling Back Changes

If something goes wrong:

```bash
# List generations
darwin-rebuild --list-generations

# Roll back to the previous generation
darwin-rebuild switch --flake . --rollback

# Or roll back to a specific generation
darwin-rebuild switch --flake . --generation X
```

## Troubleshooting

### Common Issues

1. **Permission errors during installation**:
   - Ensure you have administrator privileges
   - Run `sudo chown -R $(whoami):staff /nix` if necessary

2. **Homebrew packages not installing**:
   - Check if Homebrew is installed: `which brew`
   - Try running `brew doctor` to diagnose issues

3. **Touch ID not working for sudo**:
   - Ensure `security.pam.enableSudoTouchIdAuth = true;` is set
   - Check that `/etc/pam.d/sudo` has the TouchID entry

4. **Command not found after installation**:
   - Source your shell profile: `source ~/.zshrc`
   - Restart your terminal or shell session

5. **Error building certain packages**:
   - Use the verbose flag: `darwin-rebuild switch -v --flake .`
   - Check logs for specific error messages

### Debugging Tips

- Build without applying changes to check for errors:
  ```bash
  darwin-rebuild build --flake .
  ```

- Check system settings:
  ```bash
  defaults read
  ```

- View detailed logs:
  ```bash
  nix log /nix/store/hash-darwin-system
  ```

## Migration from Other Tools

### From Homebrew to nix-darwin

1. **List your current Homebrew packages**:
   ```bash
   brew list --formula > brew-formulas.txt
   brew list --cask > brew-casks.txt
   ```

2. **Add these to your nix-darwin config**:
   - Use `environment.systemPackages` for CLI tools
   - Use `homebrew.casks` for GUI applications

3. **Apply your configuration** and verify everything works as expected

4. **Optional: Clean up Homebrew** once you've migrated:
   ```bash
   brew remove --force $(brew list --formula) --ignore-dependencies
   ```

### From Manual Configuration to nix-darwin

1. **Audit your current setup**:
   - List apps in `/Applications`
   - Document important settings and preferences

2. **Check current defaults**:
   ```bash
   defaults read > current-defaults.txt
   ```

3. **Incrementally convert these settings** to nix-darwin format

## Relation to Other Concepts

- **[[What is Nix]]**: The foundation of the nix-darwin system
- **[[Nix Flakes Introduction]]**: Modern approach to managing nix-darwin configurations
- **[[Home Manager Configuration]]**: Complement nix-darwin with user-specific settings
- **[[Basic Nix Commands]]**: Essential commands for working with your system
- **[[Nix Development Workflows]]**: Setting up development environments

## Next Steps

→ Configure [[Home Manager Configuration]] for user-specific settings
→ Explore [[Nix Development Workflows]] for project-specific environments
→ Learn about [[Working with Nix Packages]] for advanced package management

---
Tags: #nix #nix-darwin #macos #system-configuration 