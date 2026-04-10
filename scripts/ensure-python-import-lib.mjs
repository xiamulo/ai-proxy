import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

if (process.platform !== "win32") {
  process.exit(0);
}

const pythonRoot = path.resolve("src-tauri", "pyembed", "python");
const pythonExe = path.join(pythonRoot, "python.exe");
const pythonDll = path.join(pythonRoot, "python313.dll");
const libDir = path.join(pythonRoot, "libs");
const importLib = path.join(libDir, "python313.lib");

if (!fs.existsSync(pythonExe)) {
  throw new Error(`Missing embedded Python executable: ${pythonExe}`);
}

if (fs.existsSync(importLib)) {
  console.log(`python import library already exists: ${importLib}`);
  process.exit(0);
}

if (!fs.existsSync(pythonDll)) {
  throw new Error(`Missing embedded Python DLL: ${pythonDll}`);
}

const vswhere = "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe";

if (!fs.existsSync(vswhere)) {
  throw new Error(`vswhere.exe not found: ${vswhere}`);
}

const findVisualStudioTool = (toolName) => {
  const output = execFileSync(
    vswhere,
    [
      "-latest",
      "-products",
      "*",
      "-requires",
      "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
      "-find",
      `VC\\Tools\\MSVC\\**\\bin\\Hostx64\\x64\\${toolName}`,
    ],
    { encoding: "utf8" },
  )
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find(Boolean);

  if (!output) {
    throw new Error(`Unable to locate ${toolName} via vswhere`);
  }

  return output;
};

const dumpbin = findVisualStudioTool("dumpbin.exe");
const libExe = findVisualStudioTool("lib.exe");

const dumpOutput = execFileSync(dumpbin, ["/exports", pythonDll], {
  encoding: "utf8",
  maxBuffer: 20 * 1024 * 1024,
});

const exportNames = dumpOutput
  .split(/\r?\n/)
  .map((line) => line.match(/^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+([A-Za-z_][A-Za-z0-9_@?]*)\s*$/))
  .filter(Boolean)
  .map((match) => match[1]);

if (exportNames.length === 0) {
  throw new Error(`No exports parsed from ${pythonDll}`);
}

fs.mkdirSync(libDir, { recursive: true });

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "mtga-python-lib-"));
const defPath = path.join(tempDir, "python313.def");

const defContents = [
  "LIBRARY python313",
  "EXPORTS",
  ...exportNames.map((name) => `  ${name}`),
].join(os.EOL);

fs.writeFileSync(defPath, `${defContents}${os.EOL}`, "utf8");

try {
  execFileSync(libExe, [`/def:${defPath}`, "/machine:x64", `/out:${importLib}`], {
    stdio: "inherit",
  });
} finally {
  fs.rmSync(tempDir, { recursive: true, force: true });
}

console.log(`generated python import library: ${importLib}`);
