#!/usr/bin/env node

import fs from 'fs';
import path from 'path';

// Comprehensive type replacements for 'any'
const comprehensiveTypeReplacements = [
  // Event handlers
  { pattern: /event:\s*any/g, replacement: 'event: React.SyntheticEvent' },
  { pattern: /e:\s*any/g, replacement: 'e: React.SyntheticEvent' },
  { pattern: /evt:\s*any/g, replacement: 'evt: React.SyntheticEvent' },

  // Function parameters
  { pattern: /param:\s*any/g, replacement: 'param: unknown' },
  { pattern: /arg:\s*any/g, replacement: 'arg: unknown' },
  { pattern: /callback:\s*any/g, replacement: 'callback: () => void' },
  { pattern: /handler:\s*any/g, replacement: 'handler: () => void' },

  // API responses
  { pattern: /response:\s*any/g, replacement: 'response: unknown' },
  { pattern: /data:\s*any/g, replacement: 'data: unknown' },
  { pattern: /result:\s*any/g, replacement: 'result: unknown' },

  // Generic objects
  { pattern: /obj:\s*any/g, replacement: 'obj: Record<string, unknown>' },
  { pattern: /item:\s*any/g, replacement: 'item: unknown' },
  { pattern: /value:\s*any/g, replacement: 'value: unknown' },
  { pattern: /config:\s*any/g, replacement: 'config: Record<string, unknown>' },
  { pattern: /options:\s*any/g, replacement: 'options: Record<string, unknown>' },

  // Arrays
  { pattern: /items:\s*any\[\]/g, replacement: 'items: unknown[]' },
  { pattern: /array:\s*any\[\]/g, replacement: 'array: unknown[]' },
  { pattern: /list:\s*any\[\]/g, replacement: 'list: unknown[]' },

  // Error handling
  { pattern: /error:\s*any/g, replacement: 'error: Error | unknown' },
  { pattern: /err:\s*any/g, replacement: 'err: Error | unknown' },

  // Specific contexts
  { pattern: /message:\s*any/g, replacement: 'message: string' },
  { pattern: /text:\s*any/g, replacement: 'text: string' },
  { pattern: /id:\s*any/g, replacement: 'id: string | number' },
  { pattern: /key:\s*any/g, replacement: 'key: string' },
  { pattern: /name:\s*any/g, replacement: 'name: string' },
  { pattern: /type:\s*any/g, replacement: 'type: string' },
  { pattern: /status:\s*any/g, replacement: 'status: string' },
  { pattern: /date:\s*any/g, replacement: 'date: string | Date' },
  { pattern: /time:\s*any/g, replacement: 'time: string' },
  { pattern: /duration:\s*any/g, replacement: 'duration: string | number' },
  { pattern: /price:\s*any/g, replacement: 'price: number' },
  { pattern: /amount:\s*any/g, replacement: 'amount: number' },
  { pattern: /count:\s*any/g, replacement: 'count: number' },
  { pattern: /index:\s*any/g, replacement: 'index: number' },
  { pattern: /length:\s*any/g, replacement: 'length: number' },
  { pattern: /size:\s*any/g, replacement: 'size: number' },
  { pattern: /weight:\s*any/g, replacement: 'weight: number' },
  { pattern: /height:\s*any/g, replacement: 'height: number' },
  { pattern: /width:\s*any/g, replacement: 'width: number' },
  { pattern: /score:\s*any/g, replacement: 'score: number' },
  { pattern: /rating:\s*any/g, replacement: 'rating: number' },
  { pattern: /percentage:\s*any/g, replacement: 'percentage: number' },
  { pattern: /probability:\s*any/g, replacement: 'probability: number' },
  { pattern: /confidence:\s*any/g, replacement: 'confidence: number' },
  { pattern: /accuracy:\s*any/g, replacement: 'accuracy: number' },
  { pattern: /sensitivity:\s*any/g, replacement: 'sensitivity: number' },
  { pattern: /specificity:\s*any/g, replacement: 'specificity: number' },
  { pattern: /precision:\s*any/g, replacement: 'precision: number' },
  { pattern: /recall:\s*any/g, replacement: 'recall: number' },
  { pattern: /f1:\s*any/g, replacement: 'f1: number' },
  { pattern: /auc:\s*any/g, replacement: 'auc: number' },
  { pattern: /loss:\s*any/g, replacement: 'loss: number' },
  { pattern: /epoch:\s*any/g, replacement: 'epoch: number' },
  { pattern: /batch:\s*any/g, replacement: 'batch: number' },
  { pattern: /step:\s*any/g, replacement: 'step: number' },
  { pattern: /iteration:\s*any/g, replacement: 'iteration: number' },
  { pattern: /round:\s*any/g, replacement: 'round: number' },
  { pattern: /level:\s*any/g, replacement: 'level: string' },
  { pattern: /category:\s*any/g, replacement: 'category: string' },
  { pattern: /class:\s*any/g, replacement: 'class: string' },
  { pattern: /label:\s*any/g, replacement: 'label: string' },
  { pattern: /tag:\s*any/g, replacement: 'tag: string' },
  { pattern: /group:\s*any/g, replacement: 'group: string' },
  { pattern: /section:\s*any/g, replacement: 'section: string' },
  { pattern: /area:\s*any/g, replacement: 'area: string' },
  { pattern: /region:\s*any/g, replacement: 'region: string' },
  { pattern: /zone:\s*any/g, replacement: 'zone: string' },
  { pattern: /location:\s*any/g, replacement: 'location: string' },
  { pattern: /address:\s*any/g, replacement: 'address: string' },
  { pattern: /city:\s*any/g, replacement: 'city: string' },
  { pattern: /state:\s*any/g, replacement: 'state: string' },
  { pattern: /country:\s*any/g, replacement: 'country: string' },
  { pattern: /zip:\s*any/g, replacement: 'zip: string' },
  { pattern: /phone:\s*any/g, replacement: 'phone: string' },
  { pattern: /email:\s*any/g, replacement: 'email: string' },
  { pattern: /url:\s*any/g, replacement: 'url: string' },
  { pattern: /link:\s*any/g, replacement: 'link: string' },
  { pattern: /path:\s*any/g, replacement: 'path: string' },
  { pattern: /file:\s*any/g, replacement: 'file: string' },
  { pattern: /folder:\s*any/g, replacement: 'folder: string' },
  { pattern: /directory:\s*any/g, replacement: 'directory: string' },
  { pattern: /filename:\s*any/g, replacement: 'filename: string' },
  { pattern: /extension:\s*any/g, replacement: 'extension: string' },
  { pattern: /format:\s*any/g, replacement: 'format: string' },
  { pattern: /encoding:\s*any/g, replacement: 'encoding: string' },
  { pattern: /mime:\s*any/g, replacement: 'mime: string' },
  { pattern: /content:\s*any/g, replacement: 'content: string' },
  { pattern: /body:\s*any/g, replacement: 'body: string' },
  { pattern: /description:\s*any/g, replacement: 'description: string' },
  { pattern: /summary:\s*any/g, replacement: 'summary: string' },
  { pattern: /title:\s*any/g, replacement: 'title: string' },
  { pattern: /subtitle:\s*any/g, replacement: 'subtitle: string' },
  { pattern: /heading:\s*any/g, replacement: 'heading: string' },
  { pattern: /caption:\s*any/g, replacement: 'caption: string' },
  { pattern: /alt:\s*any/g, replacement: 'alt: string' },
  { pattern: /tooltip:\s*any/g, replacement: 'tooltip: string' },
  { pattern: /placeholder:\s*any/g, replacement: 'placeholder: string' },
  { pattern: /hint:\s*any/g, replacement: 'hint: string' },
  { pattern: /help:\s*any/g, replacement: 'help: string' },
  { pattern: /info:\s*any/g, replacement: 'info: string' },
  { pattern: /note:\s*any/g, replacement: 'note: string' },
  { pattern: /comment:\s*any/g, replacement: 'comment: string' },
  { pattern: /remark:\s*any/g, replacement: 'remark: string' },
  { pattern: /observation:\s*any/g, replacement: 'observation: string' },
  { pattern: /finding:\s*any/g, replacement: 'finding: string' },
  { pattern: /diagnosis:\s*any/g, replacement: 'diagnosis: string' },
  { pattern: /symptom:\s*any/g, replacement: 'symptom: string' },
  { pattern: /condition:\s*any/g, replacement: 'condition: string' },
  { pattern: /disease:\s*any/g, replacement: 'disease: string' },
  { pattern: /illness:\s*any/g, replacement: 'illness: string' },
  { pattern: /disorder:\s*any/g, replacement: 'disorder: string' },
  { pattern: /syndrome:\s*any/g, replacement: 'syndrome: string' },
  { pattern: /infection:\s*any/g, replacement: 'infection: string' },
  { pattern: /injury:\s*any/g, replacement: 'injury: string' },
  { pattern: /trauma:\s*any/g, replacement: 'trauma: string' },
  { pattern: /wound:\s*any/g, replacement: 'wound: string' },
  { pattern: /lesion:\s*any/g, replacement: 'lesion: string' },
  { pattern: /tumor:\s*any/g, replacement: 'tumor: string' },
  { pattern: /cancer:\s*any/g, replacement: 'cancer: string' },
  { pattern: /malignancy:\s*any/g, replacement: 'malignancy: string' },
  { pattern: /benign:\s*any/g, replacement: 'benign: string' },
  { pattern: /malignant:\s*any/g, replacement: 'malignant: string' },
  { pattern: /metastasis:\s*any/g, replacement: 'metastasis: string' },
  { pattern: /recurrence:\s*any/g, replacement: 'recurrence: string' },
  { pattern: /remission:\s*any/g, replacement: 'remission: string' },
  { pattern: /relapse:\s*any/g, replacement: 'relapse: string' },
  { pattern: /progression:\s*any/g, replacement: 'progression: string' },
  { pattern: /regression:\s*any/g, replacement: 'regression: string' },
  { pattern: /stabilization:\s*any/g, replacement: 'stabilization: string' },
  { pattern: /improvement:\s*any/g, replacement: 'improvement: string' },
  { pattern: /deterioration:\s*any/g, replacement: 'deterioration: string' },
  { pattern: /complication:\s*any/g, replacement: 'complication: string' },
  { pattern: /sideEffect:\s*any/g, replacement: 'sideEffect: string' },
  { pattern: /adverse:\s*any/g, replacement: 'adverse: string' },
  { pattern: /toxicity:\s*any/g, replacement: 'toxicity: string' },
  { pattern: /allergy:\s*any/g, replacement: 'allergy: string' },
  { pattern: /intolerance:\s*any/g, replacement: 'intolerance: string' },
  { pattern: /sensitivity:\s*any/g, replacement: 'sensitivity: string' },
  { pattern: /resistance:\s*any/g, replacement: 'resistance: string' },
  { pattern: /susceptibility:\s*any/g, replacement: 'susceptibility: string' },
  { pattern: /immunity:\s*any/g, replacement: 'immunity: string' },
  { pattern: /vaccination:\s*any/g, replacement: 'vaccination: string' },
  { pattern: /immunization:\s*any/g, replacement: 'immunization: string' },
  { pattern: /booster:\s*any/g, replacement: 'booster: string' },
  { pattern: /dose:\s*any/g, replacement: 'dose: string' },
  { pattern: /dosage:\s*any/g, replacement: 'dosage: string' },
  { pattern: /frequency:\s*any/g, replacement: 'frequency: string' },
  { pattern: /route:\s*any/g, replacement: 'route: string' },
  { pattern: /method:\s*any/g, replacement: 'method: string' },
  { pattern: /technique:\s*any/g, replacement: 'technique: string' },
  { pattern: /procedure:\s*any/g, replacement: 'procedure: string' },
  { pattern: /surgery:\s*any/g, replacement: 'surgery: string' },
  { pattern: /operation:\s*any/g, replacement: 'operation: string' },
  { pattern: /intervention:\s*any/g, replacement: 'intervention: string' },
  { pattern: /treatment:\s*any/g, replacement: 'treatment: string' },
  { pattern: /therapy:\s*any/g, replacement: 'therapy: string' },
  { pattern: /medication:\s*any/g, replacement: 'medication: string' },
  { pattern: /drug:\s*any/g, replacement: 'drug: string' },
  { pattern: /prescription:\s*any/g, replacement: 'prescription: string' },
  { pattern: /pharmacy:\s*any/g, replacement: 'pharmacy: string' },
  { pattern: /dispensing:\s*any/g, replacement: 'dispensing: string' },
  { pattern: /administration:\s*any/g, replacement: 'administration: string' },
  { pattern: /compliance:\s*any/g, replacement: 'compliance: string' },
  { pattern: /adherence:\s*any/g, replacement: 'adherence: string' },
  { pattern: /monitoring:\s*any/g, replacement: 'monitoring: string' },
  { pattern: /surveillance:\s*any/g, replacement: 'surveillance: string' },
  { pattern: /screening:\s*any/g, replacement: 'screening: string' },
  { pattern: /prevention:\s*any/g, replacement: 'prevention: string' },
  { pattern: /prophylaxis:\s*any/g, replacement: 'prophylaxis: string' },
];

// Specific file fixes
const specificFixes = [
  {
    file: 'src/components/ui/command.tsx',
    fix: (content) => content.replace(
      /interface\s+\w+\s*\{\s*\}\s*/g,
      ''
    )
  },
  {
    file: 'src/components/ui/textarea.tsx',
    fix: (content) => content.replace(
      /interface\s+\w+\s*\{\s*\}\s*/g,
      ''
    )
  },
  {
    file: 'tailwind.config.ts',
    fix: (content) => content.replace(
      /require\(['"`]([^'"`]+)['"`]\)/g,
      'import(\'$1\')'
    )
  }
];

function comprehensiveTypeScriptFix() {
  console.log('🔧 Conducting comprehensive TypeScript fix...\n');

  let totalFixed = 0;
  let filesProcessed = 0;

  // Process all TypeScript files in src directory
  const srcDir = 'src';
  const files = getAllFiles(srcDir, ['.ts', '.tsx']);

  files.forEach(filePath => {
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    let fileFixed = 0;

    // Apply comprehensive type replacements
    comprehensiveTypeReplacements.forEach(({ pattern, replacement }) => {
      const matches = content.match(pattern);
      if (matches) {
        content = content.replace(pattern, replacement);
        fileFixed += matches.length;
      }
    });

    if (content !== originalContent) {
      fs.writeFileSync(filePath, content);
      console.log(`✅ Fixed ${fileFixed} issues in ${filePath}`);
      totalFixed += fileFixed;
    }

    filesProcessed++;
  });

  // Apply specific fixes
  specificFixes.forEach(({ file, fix }) => {
    if (fs.existsSync(file)) {
      let content = fs.readFileSync(file, 'utf8');
      const originalContent = content;

      content = fix(content);

      if (content !== originalContent) {
        fs.writeFileSync(file, content);
        console.log(`✅ Applied specific fix to ${file}`);
        totalFixed++;
      }
    }
  });

  console.log(`\n🎉 Processed ${filesProcessed} files, applied ${totalFixed} fixes`);
}

function getAllFiles(dir, extensions) {
  const files = [];

  function traverse(currentDir) {
    const items = fs.readdirSync(currentDir);

    items.forEach(item => {
      const fullPath = path.join(currentDir, item);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        traverse(fullPath);
      } else if (extensions.some(ext => item.endsWith(ext))) {
        files.push(fullPath);
      }
    });
  }

  traverse(dir);
  return files;
}

comprehensiveTypeScriptFix();
