{ nixpkgsPath }:

let
  nixpkgs = builtins.toPath nixpkgsPath;
  pkgs =
    if builtins.pathExists (nixpkgs + "/default.nix") then
      import nixpkgs { }
    else if builtins.pathExists (nixpkgs + "/pkgs/top-level/all-packages.nix") then
      import (nixpkgs + "/pkgs/top-level/all-packages.nix") {
        system = builtins.currentSystem;
      }
    else if builtins.pathExists (nixpkgs + "/pkgs/system/all-packages.nix") then
      import (nixpkgs + "/pkgs/system/all-packages.nix") {
        system = builtins.currentSystem;
      }
    else
      throw "this Nixpkgs revision has no supported package-set entry point";
  packageNames = builtins.attrNames pkgs;
  isFetcher = name: builtins.match "fetch.*" name != null;
in
builtins.filter isFetcher packageNames
