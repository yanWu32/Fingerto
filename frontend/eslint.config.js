import js from '@eslint/js'
import vue from 'eslint-plugin-vue'
import tseslint from 'typescript-eslint'

// ESLint 扁平配置（规范 5.1 工程化 + 4.2 代码规范）
export default tseslint.config(
  {
    ignores: ['dist', 'node_modules', '*.config.ts'],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...vue.configs['flat/recommended'],
  {
    // <script setup lang="ts"> 需要 TS 解析器，否则 eslint 默认用 espree 解析会报
    // “Parsing error: Unexpected token {”（import type { ... } 语法）
    files: ['**/*.vue'],
    languageOptions: {
      parserOptions: {
        parser: tseslint.parser,
        ecmaVersion: 'latest',
        sourceType: 'module',
      },
    },
  },
  {
    // 演示页需要把 prompt / 输出文本中的关键 token 包 <mark> 高亮，只能用 v-html。
    // 安全性由 utils/sanitize.ts 的 highlightTokens 保证：先转义 HTML 特殊字符、
    // 再插入自有标签，且有 sanitize.test.ts 的 XSS 用例覆盖（规范 7.1 仍满足）。
    files: ['src/views/DemoView.vue'],
    rules: {
      'vue/no-v-html': 'off',
    },
  },
  {
    rules: {
      'vue/multi-word-component-names': 'off',
      // 下面两条是纯排版规则，与 Prettier 的换行策略直接冲突
      // （eslint --fix 与 prettier 会互相打架）：排版统一交给 Prettier 负责。
      'vue/max-attributes-per-line': 'off',
      'vue/singleline-html-element-content-newline': 'off',
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
    },
  },
)
