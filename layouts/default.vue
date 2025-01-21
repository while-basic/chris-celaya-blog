<template>
  <div>
    <slot />
    <Analytics />
  </div>
</template>

<script setup lang="ts">
import Analytics from '~/components/Analytics.vue'
</script> 