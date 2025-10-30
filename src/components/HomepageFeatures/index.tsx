import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: '零門檻入門',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        使用 Google Colab 或家用 GPU 即可開始，無需昂貴的硬體設備。
        完整的中文教學，讓你快速上手 AI 模型訓練。
      </>
    ),
  },
  {
    title: '實戰導向',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        每個章節都提供可直接執行的程式碼範例，從資料準備到模型部署，
        涵蓋完整流程，讓你學以致用。
      </>
    ),
  },
  {
    title: '專業領域應用',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        學習如何針對特定領域訓練專屬的 AI 模型，打造真正符合需求的
        智慧應用，提升工作效率。
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
