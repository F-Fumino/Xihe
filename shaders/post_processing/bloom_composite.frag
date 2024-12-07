#version 450
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(binding = 0) uniform sampler2D originalImage;
layout(binding = 1) uniform sampler2D blurredImage;

layout(push_constant) uniform PushConstants {
    float intensity;      // ����ǿ��
    float tint_r;        // ����ɫ�� R
    float tint_g;        // ����ɫ�� G
    float tint_b;        // ����ɫ�� B
    float saturation;    // ���ⱥ�Ͷ�
} pc;

// �������Ͷ�
vec3 adjustSaturation(vec3 color, float saturation) {
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luminance), color, saturation);
}

// ACES tone mapping
vec3 ACESFilm(vec3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// �򵥵�Reinhard tone mapping
vec3 Reinhard(vec3 color) {
    return color / (1.0 + color);
}

void main() {
    // ����ԭʼͼ���ģ����ķ���ͼ��
    vec3 originalColor = texture(originalImage, inUV).rgb;
    vec3 bloomColor = texture(blurredImage, inUV).rgb;
    
    // ������
    bloomColor = adjustSaturation(bloomColor, pc.saturation);
    bloomColor *= vec3(pc.tint_r, pc.tint_g, pc.tint_b);
    
    // ���
    vec3 finalColor = originalColor + bloomColor * pc.intensity;
    
    // Ӧ��tone mapping
    // finalColor = ACESFilm(finalColor);  // ����ʹ�� Reinhard(finalColor)

    finalColor = finalColor / (1.0 + finalColor);
    
    outColor = vec4(finalColor, 1.0);
}